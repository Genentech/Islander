import time, torch, torch.nn as nn, torch.nn.functional as F
import Utils_Handler as uh

# from scLoss import HSICLoss, DDCLoss, KCCALoss, InfoNCELoss


class AutoEncoder(nn.Module):
    def __init__(
        self,
        n_gene: int,
        bn_eps=1e-05,
        batchnorm=True,
        bn_momentum=0.1,
        dropout_layer=False,
        mlp_size=[128, 128],
        reconstruct_loss=nn.MSELoss(),
        **kwargs,
    ):
        super().__init__()

        self.n_gene = n_gene
        self.bn_eps = bn_eps
        self.mlp_size = mlp_size
        self.batchnorm = batchnorm
        self.bn_momentum = bn_momentum
        self.dropout_layer = dropout_layer
        self.reconstruct_loss = reconstruct_loss

        self.encoder = self._encode()
        self.decoder = self._decode()

    def _batchnorm1d(self, size):
        # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        # Input shape here is (N, C), (# Cell, # Gene)
        # default batch1d: BatchNorm1d(64, eps=1e-05, momentum=0.1)
        # scvi batch1d: BatchNorm1d(64, eps=1e-03, momentum=0.01)
        return nn.BatchNorm1d(size, momentum=self.bn_momentum, eps=self.bn_eps)

    def _encode(self):
        # on the order of functional layers
        # ref: https://stackoverflow.com/questions/39691902
        # Input -> FC -> BN -> ReLu (-> Dropout) -> FC -> Reconstruction

        enc = []
        enc.append(nn.Linear(self.n_gene, self.mlp_size[0]))
        if self.batchnorm:
            enc.append(self._batchnorm1d(self.mlp_size[0]))
        enc.append(nn.ReLU())
        if self.dropout_layer:
            enc.append(nn.Dropout(p=0.1))

        mid_idx = len(self.mlp_size) // 2 - (len(self.mlp_size) + 1) % 2
        for i in range(mid_idx):
            enc.append(nn.Linear(self.mlp_size[i], self.mlp_size[i + 1]))
            if self.batchnorm:
                enc.append(self._batchnorm1d(self.mlp_size[i + 1]))
            enc.append(nn.ReLU())
            if self.dropout_layer:
                enc.append(nn.Dropout(p=0.1))

        # NOTE: make sure the last layer is linear
        enc.pop()
        if self.batchnorm:
            enc.pop()
        if self.dropout_layer:
            enc.pop()
        return nn.Sequential(*enc)

    def _decode(self):
        dec = []
        mid_ = len(self.mlp_size) // 2 - (len(self.mlp_size) + 1) % 2

        for i in range(len(self.mlp_size) // 2):
            dec.append(nn.Linear(self.mlp_size[mid_ + i], self.mlp_size[mid_ + i + 1]))
            if self.batchnorm:
                dec.append(self._batchnorm1d(self.mlp_size[mid_ + i + 1]))
            dec.append(nn.ReLU())
        dec.append(nn.Linear(self.mlp_size[-1], self.n_gene))
        dec.append(nn.ReLU())  # norm counts are always non-negative
        return nn.Sequential(*dec)

    def forward(self, counts_: torch.Tensor, **kwargs) -> torch.Tensor:
        emb_ = self.encoder(counts_)  # (# Cell, # Embedding)
        out_ = self.decoder(emb_)
        return out_

    def extra_repr(self, x):
        return self.encoder(x)

    def recon_repr(self, x):
        return self.decoder(x)

    def loss_(self, rec_pred: torch.Tensor, rec_target: torch.Tensor, **kwargs):
        return self.reconstruct_loss(rec_pred, rec_target)

    def norm_inf(self, rec_):
        normed_counts = torch.exp(rec_) - 1
        size_factors = 1e4 / normed_counts.sum(axis=1)
        rec_new = torch.log(normed_counts * size_factors[:, None] + 1)
        return rec_new


class AE_Concept(AutoEncoder):
    def __init__(
        self,
        n_gene: int,
        n_concept=0,
        leak_dim=16,
        cell2cat=None,
        bn_eps=1e-05,
        bn_momentum=0.1,
        batchnorm=True,
        mlp_size=[64, 64],
        dropout_layer=False,
        with_projector=True,
        **kwargs,
    ) -> None:
        super().__init__(
            n_gene=n_gene,
            bn_eps=bn_eps,
            mlp_size=mlp_size,
            batchnorm=batchnorm,
            bn_momentum=bn_momentum,
            dropout_layer=dropout_layer,
        )
        self.leak_dim = leak_dim
        self.cell2cat = cell2cat
        self.n_concept = n_concept

        if len(self.mlp_size) % 2 == 0:
            self.mlp_size.insert(len(self.mlp_size) // 2, n_concept + leak_dim)

        print("=" * 28, " Model architecture ", "=" * 27)
        print("Leak dim:", self.leak_dim)
        print("Dropout:", self.dropout_layer)
        print("MLP size:", self.mlp_size)
        print("Batchnorm:", batchnorm, ", momentum:", bn_momentum, ", eps:", bn_eps)
        print("=" * 77)

        self.encoder = self._encode()
        self.decoder = self._decode()
        if with_projector:
            self.projector = self._proj()

    def _proj(self):
        NUM_CELLTYPE = self.cell2cat.__len__()
        return nn.Linear(self.leak_dim, NUM_CELLTYPE)
        # return nn.Linear(self.leak_dim, NUM_CELLTYPE, bias=False)

    def _loss_rec(self, pred: torch.Tensor, target: torch.Tensor):
        """Reconstruction Loss"""
        return self.reconstruct_loss(pred, target)

    def _loss_celltype(self, emb_b: torch.Tensor, cell_type: torch.Tensor):
        """Cell Type Cross Entropy Loss"""
        emb_b = self.projector(emb_b)
        emb_b = emb_b.log_softmax(dim=-1)
        return F.nll_loss(emb_b, cell_type.to(emb_b.device))

    def forward(self, counts_: torch.Tensor, **kwargs) -> torch.Tensor:
        emb_ = self.encoder(counts_)
        out_ = self.decoder(emb_)
        return out_

    def acc_celltype_(self, emb_b: torch.Tensor, cell_type: torch.Tensor):
        self.projector.eval()
        emb_b = self.projector(emb_b)
        emb_b = emb_b.softmax(dim=-1).max(-1)[1]
        return emb_b.eq(cell_type.to(emb_b.device)).mean(dtype=float).cpu().item()


Model_ZOO = {
    "base-ae": AutoEncoder,
    "ae-concept": AE_Concept,
}

if __name__ == "__main__":
    #
    import argparse
    from scDataset import scDataset, collate_fn
    from torch.utils.data import DataLoader

    #
    LR = 0.001
    N_EPOCHS = 3
    MINCELLS = 256
    DEVICE = torch.device("cuda:3")

    for DATASET in [
        "lung",
        "lung_fetal_donor",
        "lung_fetal_organoid",
        "brain",
        "breast",
        "heart",
        "eye",
        "gut_fetal",
        "skin",
        "COVID",
        "pancreas",
    ]:
        # scData_Train = scDataset(dataset=DATASET, training=True, rm_cache=True)
        scData_Train = scDataset(dataset=DATASET, training=True)
        scData_Test = scDataset(dataset=DATASET, verbose=False, training=False)
        scTrainLoader = DataLoader(
            scData_Train,
            batch_size=1,
            shuffle=True,
            num_workers=8,
            collate_fn=collate_fn,
        )
        scTestLoader = DataLoader(
            scData_Test,
            batch_size=1,
            shuffle=True,
            num_workers=8,
            collate_fn=collate_fn,
        )
        n_genes = scData_Train.n_vars

        def ArgParser():
            parser = argparse.ArgumentParser()
            parser.add_argument("--leak_dim", type=int, default=16)
            parser.add_argument(
                "--mlp_size", type=int, nargs="+", default=[128, 128]
            )  # default=[128, 128, 128, 128]
            return parser.parse_args()

        args = ArgParser()
        MODEL = AE_Concept(
            n_gene=n_genes,
            dropout_layer=True,
            mlp_size=args.mlp_size,
            leak_dim=args.leak_dim,
            cell2cat=scData_Train.CELL2CAT,
        ).to(DEVICE)

        Opt = torch.optim.Adam(MODEL.parameters(), lr=LR)
        start = time.time()
        for epoch in range(N_EPOCHS):
            # LossWeights = {
            #     "reconstruction": [1.0],
            #     "celltype": [],
            #     "concept": [],
            # }
            # print("Epoch: %2d" % epoch)
            for id_, batch_ in enumerate(scTrainLoader):
                #
                Opt.zero_grad()
                counts = batch_["counts"].squeeze().to(DEVICE)
                #
                rec_ = MODEL(counts)
                emb_ = MODEL.extra_repr(counts)
                loss_rec = MODEL._loss_rec(counts, rec_)
                loss_celltype = MODEL._loss_celltype(emb_, batch_["cell_type"])

                # LossWeights["celltype"].append(
                #     uh.Weights_on_GradNorm(AE_CBLEAKAGE, loss_rec, loss_celltype)
                # )
                loss = loss_rec + 1e-2 * loss_celltype
                loss.backward()
                Opt.step()
            # for key, value in LossWeights.items():
            #     LossWeights[key] = uh.mean_(value)
            #     print("%17s, %.6f" % (key, LossWeights[key]))
            with torch.no_grad():
                acc_ = []
                for _, batch_ in enumerate(scTestLoader):
                    counts = batch_["counts"].to(DEVICE)
                    acc_.append(
                        MODEL.acc_celltype_(
                            emb_b=MODEL.extra_repr(counts),
                            cell_type=batch_["cell_type"],
                        )
                    )
                print(
                    "Epoch: %2d, Test Acc: %.2f, EST: %.1f Mins"
                    % (epoch, 100 * uh.mean_(acc_), (time.time() - start) // 60)
                )
