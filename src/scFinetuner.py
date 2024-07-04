# https://docs.scvi-tools.org/en/stable/tutorials/notebooks/harmonization.html
# https://scib-metrics.readthedocs.io/en/stable/notebooks/lung_example.html

import os, torch, json, time, wandb, shutil, scanpy as sc, numpy as np, torch.nn as nn, torch.nn.functional as F, Utils_Handler as uh, Data_Handler as dh, scLoss as scL, scDataset as scD
from torch.optim.lr_scheduler import CosineAnnealingLR
from scLoss import HSICLoss, InfoNCELoss
from torch.utils.data import DataLoader
from ArgParser import Parser_Finetuner
from os.path import join, exists
from scBenchmarker import scIB
from torch.optim import Adam
from copy import deepcopy

# set_seed(dh.SEED)

PATIENCE = 10
MIX_UP = False
REC_GRADS = False
# CONSTANT_LR = True
CONSTANT_LR = False
_n = dh.TOTAL_CONCEPTS


# 25142 Genes, Limb Intersection


class scFineTuner(scIB):
    """Basic Cross Entropy Fine Tuner"""

    def __init__(self, args):
        super().__init__(args=args)

    def _load_adata_(self):
        assert self.args.dataset != "hlca"

        if self.args.use_raw:
            adata = sc.read(dh.DATA_RAW_[self.args.dataset], first_column_names=True)
            if self.args.highvar:
                self._str_formatter("using top %d highly variable genes" % self.args.highvar_n)
                sc.pp.highly_variable_genes(
                    adata,
                    subset=True,
                    flavor="seurat_v3",
                    batch_key=self.batch_key,
                    n_top_genes=self.args.highvar_n,
                )
        else:
            _suffix = "_hvg" if self.args.highvar else ""
            adata = sc.read(dh.DATA_EMB_[self.args.dataset + _suffix])

        if self.args.batch_id is not None:
            adata = adata[adata.obs[self.batch_key].isin(self.args.batch_id), :]
        self.adata = adata

        return

    def _prep_dataloader_(self):
        self._str_formatter("Dataloader, Fine-Tuning")
        _, gene_pad = self.args.dataset.split("_")
        self._dataset = scD.scDataset_Transfer(
            adata=self.adata,
            prefix="Finetune_" + self.args.dataset,
            n_cells=self.args.n_cells,
            dataset=self.args.dataset,
            gene_pad=gene_pad,
            with_meta=True,
            rm_cache=False,
            one_shot=False,
            few_shots=False,
            all_shots=True,
            shuffle=True,
        )

        self.n_gene = self._dataset.n_vars
        self.CONCEPTS = self._dataset.CONCEPTS
        self.NUM_CELLTYPE = len(self._dataset.CELL2CAT)
        self.scDataLoader = DataLoader(
            self._dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=scD.collate_fn,
        )
        return

    def _loss_cet(self, emb_b: torch.Tensor, cell_type: torch.Tensor):
        # cell type prediction
        emb_b = self.model.projector(emb_b)
        emb_b = emb_b.log_softmax(dim=-1)
        return F.nll_loss(emb_b, cell_type.to(emb_b.device))

    def _loss_mixup(self, emb_b, cell_type, alpha=1.0):
        # alpha=1.0, 0.2, 0.4, 0.6
        """mixed inputs, pairs of targets, and lambda"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = emb_b.size()[0]
        index = torch.randperm(batch_size)

        mixed_x = lam * emb_b + (1 - lam) * emb_b[index, :]
        y_a, y_b = cell_type, cell_type[index]

        return lam * self._loss_cet(mixed_x, y_a) + (1 - lam) * self._loss_cet(mixed_x, y_b)

    def _finetune_(self):
        model_cfg = self.args.save_path.split("/")[-1]
        dataset_cfg = self.args.save_path.split("/")[-2]
        self._str_formatter("Model: %s, Dataset: %s" % (model_cfg, dataset_cfg))
        wandb.init(
            project="Cross Entropy FT %s" % dataset_cfg,
            name=model_cfg,
            config=self.args,
        )
        self._prep_dataloader_()
        self._load_model()
        BEST_TEST = 1e4

        self.model.projector = torch.nn.Linear(self.args.leakage, self.NUM_CELLTYPE, bias=False).to(self.device)  # without bias doesn't matter that much

        if self.args.savename_ft:
            SAVE_PATH = self.args.savename_ft
        else:
            SAVE_PATH = join(dh.MODEL_DIR, uh.DATETIME)
        self._str_formatter("SAVE_PATH: %s" % SAVE_PATH)

        if exists(SAVE_PATH):
            shutil.rmtree(SAVE_PATH)
        os.makedirs(SAVE_PATH, exist_ok=True)

        with open(join(SAVE_PATH, "cfg.json"), "w") as outfile:
            json.dump(vars(self.args), outfile)

        LR = self.args.lr_ft
        NUM_EPOCHS = self.args.epoch_ft
        Opt = Adam(self.model.parameters(), lr=LR)
        LR_SCHEDULER = CosineAnnealingLR(Opt, T_max=NUM_EPOCHS)

        start = time.time()
        for epoch in range(NUM_EPOCHS):
            total_err, total_rec, total_cet, total_hsi = 0, 0, 0, 0
            cell_acc_ = []

            LossWeights = {
                "reconstruction": [1.0],
                # "infonce": [],
                "cet": [],
            }
            for batch_ in self.scDataLoader:
                Opt.zero_grad()
                counts_ = batch_["norm_counts"].squeeze().to(self.device)
                rec_ = self.model(counts_)
                emb_ = self.model.extra_repr(counts_)
                emb_b = emb_[:, dh.TOTAL_CONCEPTS :]

                loss_rec = args.w_rec_ft * self.model._loss_rec(counts_, rec_)
                loss_hsi = args.w_hsi_ft * self._loss_hsi(emb_b, batch_["meta"])
                loss_inf = args.w_inf_ft * self._loss_inf(emb_b, batch_["meta"])
                loss_cet = args.w_cet_ft * self._loss_cet(emb_b, batch_["cell_type"])

                if MIX_UP:
                    loss_cet += self._loss_mixup(emb_b, batch_["cell_type"])

                loss = loss_rec + loss_hsi + loss_inf + loss_cet
                wandb.log(
                    {
                        "Loss Reconstruction": loss_rec.item(),
                        "Loss Cell Type": loss_cet.item(),
                        "Loss InfoNCE": loss_inf.item(),
                        "Loss HSIC": loss_hsi.item(),
                    }
                )

                if REC_GRADS:
                    LossWeights["cet"].append(uh.Weights_on_GradNorm(self.model, loss_cet, loss_rec))

                else:
                    loss.backward()
                    Opt.step()

                total_err += loss.item()
                total_rec += loss_rec.item()
                total_hsi += loss_hsi.item()
                total_cet += loss_cet.item()

                cell_acc_.append(self.model.acc_celltype_(emb_b, batch_["cell_type"]))

            if not CONSTANT_LR:
                LR_SCHEDULER.step()
            lr = LR_SCHEDULER.get_last_lr()[0]
            train_err = total_err / len(self.scDataLoader)
            train_hsi = total_hsi / len(self.scDataLoader)
            train_rec = total_rec / len(self.scDataLoader)

            if uh._isnan(train_err):
                self._str_formatter("NaN Value Encountered, Quitting")
                quit()

            if epoch % 1 == 0:
                print(
                    "LR: %.6f " % lr,
                    "Epoch: %2d, " % epoch,
                    "Total Loss: %.2f, " % train_err,
                    "CellType Acc: %.2f, " % (100 * uh.mean_(cell_acc_)),
                    "EST: %.1f Mins" % ((time.time() - start) // 60),
                )
                if REC_GRADS:
                    for key, value in LossWeights.items():
                        print("%20s, %.4f" % (key, uh.mean_(value)))
                    print("\n")

            wandb.log(
                {
                    "LR": lr,
                    "Total Loss": train_err,
                    "HSIC Loss": train_hsi,
                    "Reconstruction Loss": train_rec,
                    "CellType Accuracy": 100 * uh.mean_(cell_acc_),
                }
            )

            if (epoch + 1) % 1 == 0:
                torch.save(
                    self.model.state_dict(),
                    join(SAVE_PATH, "ckpt_%d.pth" % (epoch + 1)),
                )

            if train_err < BEST_TEST:
                BEST_TEST = train_err
                torch.save(self.model.state_dict(), join(SAVE_PATH, "ckpt_best.pth"))

        return


class scFineTuner_SCL(scFineTuner):
    """Supervised Contrastive Learning"""

    def __init__(self, args):
        super().__init__(args=args)

    def _prep_dataloader_(self):
        self._str_formatter("Prep Dataloader, GeneCBM, Fine-Tuning")
        _, gene_pad = self.args.dataset.split("_")
        self._dataset = scD.scDataset_Transfer(
            adata=self.adata,
            gene_pad=gene_pad,
            prefix="Finetune_" + self.args.dataset,
            dataset=self.args.dataset,
            n_cells=self.args.n_cells,
            with_meta=True,
            rm_cache=False,
            one_shot=False,
            few_shots=False,
            all_shots=True,
            shuffle=True,
        )

        self.n_gene = self._dataset.n_vars
        self.CONCEPTS = self._dataset.CONCEPTS
        self.NUM_CELLTYPE = len(self._dataset.CELL2CAT)
        self.scDataLoader = DataLoader(
            self._dataset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            collate_fn=scD.collate_fn,
        )

        self.ContrastiveDataset = scD.ContrastiveDatasetDUAL(self._dataset)
        self.ContrastiveDatasetloader = DataLoader(
            self.ContrastiveDataset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
        )

        # self.TripletDataset = TripletDataset(self._dataset)
        # self.ContrastiveDataset = ContrastiveDataset(self._dataset)

        # self.TripletDataloader = DataLoader(
        #     self.TripletDataset,
        #     batch_size=1,
        #     shuffle=False,
        #     num_workers=4,
        #     collate_fn=collate_fn,
        # )

        return

    def _finetune_(self):
        dataset_cfg = self.args.save_path.split("/")[-2]
        model_cfg = self.args.save_path.split("/")[-1]
        wandb.init(project="SCL FT %s" % dataset_cfg, name=model_cfg, config=self.args)
        self._prep_dataloader_()
        self._load_model()

        self.model.projector = nn.Linear(self.args.leakage, self.NUM_CELLTYPE).to(self.device)

        if self.args.savename_ft:
            SAVE_PATH = self.args.savename_ft
        else:
            SAVE_PATH = join(dh.MODEL_DIR, uh.DATETIME)
        self._str_formatter("SAVE_PATH: %s" % SAVE_PATH)
        os.makedirs(SAVE_PATH, exist_ok=True)

        with open(join(SAVE_PATH, "cfg.json"), "w") as outfile:
            json.dump(vars(self.args), outfile)

        BEST_TEST = 1e4
        LR = self.args.lr_ft
        NUM_EPOCHS = self.args.epoch_ft
        Opt = Adam(self.model.parameters(), lr=LR)
        LR_SCHEDULER = CosineAnnealingLR(Opt, T_max=NUM_EPOCHS)

        LOCAL_PATIENCE = 0
        start = time.time()
        # triplet_loss = scL.TripletLoss()
        # contrastive_loss = scL.ContrastiveLoss()
        scl_loss = scL.DualBatchSupervisedContrastiveLoss()

        for epoch in range(NUM_EPOCHS):
            total_err, total_rec, total_scl = 0, 0, 0

            # for anchor, positive, negative in self.TripletDataloader:
            #     Opt.zero_grad()

            #     counts_a = anchor["norm_counts"].squeeze().to(self.device)
            #     emb_a = self.model.extra_repr(counts_a)

            #     counts_p = positive["norm_counts"].squeeze().to(self.device)
            #     emb_p = self.model.extra_repr(counts_p)

            #     counts_n = negative["norm_counts"].squeeze().to(self.device)
            #     emb_n = self.model.extra_repr(counts_n)

            #     triplet_loss = triplet_loss(emb_a, emb_p, emb_n)

            #     triplet_loss.backward()
            #     Opt.step()
            #     total_error += triplet_loss.item()

            # for batch_ in self.scDataLoader:
            #     Opt.zero_grad()
            #     counts_ = batch_["norm_counts"].squeeze().to(self.device)
            #     emb_ = self.model.extra_repr(counts_)
            #     _permute = torch.randperm(emb_.size()[0])
            #     emb_permute = emb_[_permute].to(self.device)
            #     label_ = batch_["cell_type"].to(self.device)
            #     label_permute = label_[_permute]

            #     _loss = contrastive_loss(
            #         emb_, emb_permute, label_, label_permute)
            #     _loss.backward()
            #     Opt.step()
            #     total_error += _loss.item()

            for b1, l1, b2, l2 in self.ContrastiveDatasetloader:
                Opt.zero_grad()

                """ Intra Batch """
                counts_1 = b1.squeeze().to(self.device)
                emb_1 = self.model.extra_repr(counts_1)
                l1 = torch.concat(l1).to(self.device)

                _permute = torch.randperm(emb_1.size()[0])
                emb_, l1_ = emb_1[_permute], l1[_permute]
                _loss = scl_loss(emb_1[:, _n:], emb_[:, _n:], l1, l1_)

                """ Inter Batch """
                counts_2 = b2.squeeze().to(self.device)
                emb_2 = self.model.extra_repr(counts_2)
                if len(emb_1) == len(emb_2):
                    l2 = torch.concat(l2).to(self.device)
                    _permute = torch.randperm(emb_1.size()[0])
                    emb_2, l2 = emb_2[_permute], l2[_permute]
                    _loss += scl_loss(emb_1[:, _n:], emb_2[:, _n:], l1, l2)

                _loss.backward()
                Opt.step()
                train_err += args.w_scl_ft * _loss.item()

            LR_SCHEDULER.step()
            train_err = train_err / len(self.scDataLoader)
            # train_recon_loss = total_recon / len(self.scDataLoader)

            lr = LR_SCHEDULER.get_last_lr()[0]

            if uh._isnan(train_err):
                self._str_formatter("NaN Value Encountered, Quitting")
                quit()

            if epoch % 1 == 0:
                print(
                    "LR: %.6f " % lr,
                    "Epoch: %2d, " % epoch,
                    "Total Loss: %.2f, " % train_err,
                    "EST: %.1f Mins" % ((time.time() - start) // 60),
                )

            wandb.log(
                {
                    "LR": lr,
                    "Total Loss": total_err,
                }
            )

            if (epoch + 1) % 1 == 0:
                torch.save(
                    self.model.state_dict(),
                    join(SAVE_PATH, "ckpt_%d.pth" % (epoch + 1)),
                )

            LOCAL_PATIENCE += 1
            if total_err < BEST_TEST:
                LOCAL_PATIENCE = 0
                BEST_TEST = total_err
                torch.save(self.model.state_dict(), join(SAVE_PATH, "ckpt_best.pth"))

            if LOCAL_PATIENCE > PATIENCE:
                torch.save(
                    self.model.state_dict(),
                    join(SAVE_PATH, "ckpt_%d.pth" % (epoch + 1)),
                )
                print("Patience (%d Epochs) Reached, Quitting" % PATIENCE)
                quit()

        return


MOMERY_SIZE = 65536  # The queue size in MoCo
# MOMERY_SIZE = 2048 # Smaller size does not help (overfit to positive pairs)
MOMENTUM = 0.999  # The momentum in MoCo
# TEMP = 0.07  # The temperature in MoCo


class MemoryBank(nn.Module):
    def __init__(self, input_dim, device, memory_size=MOMERY_SIZE):
        super(MemoryBank, self).__init__()

        self.memory_size = memory_size
        self.embed_bank = torch.randn(memory_size, input_dim).to(device)
        # self.embed_bank = F.normalize(self.embed_bank, dim=0).to(device)
        self.label_bank = torch.randint(0, 20, (memory_size,)).to(device)

    @torch.no_grad()
    def fetch(self, embeds_ema, labels):
        self.embed_bank = torch.concat([embeds_ema, self.embed_bank])
        self.label_bank = torch.concat([labels, self.label_bank])

        self.embed_bank = self.embed_bank[: self.memory_size, :]
        self.label_bank = self.label_bank[: self.memory_size]

        return self.embed_bank, self.label_bank

    def forward(self):
        return


class scFineTuner_PSL(scFineTuner):
    def __init__(self, args):
        super().__init__(args=args)

    def _finetune_(self):
        model_cfg = self.args.save_path.split("/")[-1]
        dataset_cfg = self.args.save_path.split("/")[-2]
        wandb.init(
            project="CFT_MB%s%s" % (dataset_cfg, self.args.trainer_ft),
            name=model_cfg,
            config=self.args,
        )

        self._prep_dataloader_()
        self._load_model()

        if self.args.savename_ft:
            SAVE_PATH = self.args.savename_ft
        else:
            SAVE_PATH = join(dh.MODEL_DIR, uh.DATETIME)
        self._str_formatter("SAVE_PATH: %s" % SAVE_PATH)

        if exists(SAVE_PATH):
            shutil.rmtree(SAVE_PATH)
        os.makedirs(SAVE_PATH, exist_ok=True)

        with open(join(SAVE_PATH, "cfg.json"), "w") as outfile:
            json.dump(vars(self.args), outfile)

        """ === Projector === """
        self.model.projector = nn.Linear(self.args.leakage, self.NUM_CELLTYPE, bias=True).to(self.device)
        # bias=False, doesn't matter that much

        """ === MoCo with Memory Bank === """
        # encoder_q and encoder_k as the query encoder and key encoder
        encoder_q = self.model.encoder
        encoder_k = deepcopy(encoder_q).to(self.device)
        # print(encoder_q is self.model.encoder) -> True
        # print(encoder_k is self.model.encoder) -> False

        encoder_k.load_state_dict(encoder_q.state_dict())
        for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        BEST_TEST = 1e4
        LR = self.args.lr_ft
        NUM_EPOCHS = self.args.epoch_ft
        Opt = Adam(self.model.parameters(), lr=LR)
        LR_SCHEDULER = CosineAnnealingLR(Opt, T_max=NUM_EPOCHS)
        if self.args.trainer_ft == "scl":
            TRAINER = scL.SCL(MemoryBank(self.model.leak_dim, self.device))
        elif self.args.trainer_ft == "simple":
            TRAINER = scL.SimPLE(MemoryBank(self.model.leak_dim, self.device))
        else:
            raise NotImplementedError

        LOCAL_PATIENCE = 0
        start = time.time()

        for epoch in range(NUM_EPOCHS):
            total_err, total_psl, total_rec = 0, 0, 0
            cell_acc_ = []
            LossWeights = {
                "reconstruction": [1.0],
                "psl": [],
            }
            for item in self.scDataLoader:
                Opt.zero_grad()

                counts_ = item["norm_counts"].to(self.device)
                labels_ = item["cell_type"].to(self.device)
                _permute = torch.randperm(counts_.size()[0])
                counts_ = counts_[_permute]
                labels_ = labels_[_permute]

                """ === MoCo with Memory Bank === """
                with torch.no_grad():
                    # update the key encoder
                    for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
                        param_k.data = param_k.data * MOMENTUM + param_q.data * (1.0 - MOMENTUM)

                    # compute key features
                    k = encoder_k(counts_)[:, dh.TOTAL_CONCEPTS :]
                    # k = F.normalize(k, dim=1)

                # compute query features
                q = encoder_q(counts_)[:, dh.TOTAL_CONCEPTS :]
                loss_pls = args.w_psl_ft * TRAINER(q, labels_, k)

                rec_ = self.model(counts_)
                loss_rec = args.w_rec_ft * nn.MSELoss()(counts_, rec_)
                loss_cet = args.w_cet_ft * self._loss_cet(q, labels_)
                loss = loss_rec + loss_pls + loss_cet

                # backprop
                total_psl += loss_pls.item()
                total_rec += loss_rec.item()
                total_err += loss.item()

                cell_acc_.append(self.model.acc_celltype_(q, labels_))

                # update / record gradients
                if REC_GRADS and (epoch + 1) % 2 == 0:
                    LossWeights["psl"].append(uh.Weights_on_GradNorm(self.model, loss_pls, loss_rec))
                else:
                    loss.backward()
                    Opt.step()

            # lr update
            if not CONSTANT_LR:
                LR_SCHEDULER.step()
            lr = LR_SCHEDULER.get_last_lr()[0]

            if uh._isnan(total_err):
                self._str_formatter("NaN Value Encountered, Quitting")
                quit()

            if epoch % 1 == 0:
                print(
                    "LR: %.6f, " % lr,
                    "Epoch: %2d, " % epoch,
                    "Total Loss: %.4f, " % (total_err / len(self.scDataLoader)),
                    "CellType Acc: %.2f, " % (100 * uh.mean_(cell_acc_)),
                    "EST: %.1f Mins" % ((time.time() - start) // 60),
                )

            wandb.log(
                {
                    "LR": lr,
                    "Total Loss": total_err / len(self.scDataLoader),
                    "Total PSL Loss": total_psl / len(self.scDataLoader),
                    "Total REC Loss": total_rec / len(self.scDataLoader),
                }
            )

            if (epoch + 1) % 1 == 0:
                torch.save(
                    self.model.state_dict(),
                    join(SAVE_PATH, "ckpt_%d.pth" % (epoch + 1)),
                )
                if REC_GRADS:
                    raise ValueError
                    # for key, value in LossWeights.items():
                    #     print("%20s, %.4f" % (key, uh.mean_(value)))
                    # print("\n")

            LOCAL_PATIENCE += 1
            if total_err < BEST_TEST:
                LOCAL_PATIENCE = 0
                BEST_TEST = total_err
                torch.save(self.model.state_dict(), join(SAVE_PATH, "ckpt_best.pth"))

            if LOCAL_PATIENCE > PATIENCE:
                torch.save(
                    self.model.state_dict(),
                    join(SAVE_PATH, "ckpt_%d.pth" % (epoch + 1)),
                )
                print("Patience (%d Epochs) Reached, Quitting" % PATIENCE)
                quit()
        return


if __name__ == "__main__":
    args = Parser_Finetuner()

    if args.mode == "ce":
        ft = scFineTuner(args)
    elif args.mode == "scl":
        ft = scFineTuner_SCL(args)
    elif args.mode == "psl":
        ft = scFineTuner_PSL(args)
    ft._finetune_()
