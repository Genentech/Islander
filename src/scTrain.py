from __future__ import absolute_import, division, print_function
import os, time, json, torch, wandb, shutil, Data_Handler as dh, numpy as np, torch.nn.functional as F
from scDataset import scDataset, ContrastiveDatasetDUAL, TripletDatasetDUAL, collate_fn
from scLoss import TripletLoss, SupervisedContrastiveLoss
from torch.optim import SGD, RMSprop, Adam, ASGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from Utils_Handler import _isnan, mean_, DATETIME
from torch.utils.data import DataLoader
from ArgParser import Parser_Trainer
from os.path import join, exists
from scModel import Model_ZOO

PATIENCE = 10
_Optimizer = {
    "sgd": SGD,
    "asgd": ASGD,
    "adam": Adam,
    "adamw": AdamW,
    "rmsprop": RMSprop,
}


class scTrainer:
    def __init__(self, args, mixup=False, **kwargs) -> None:
        self.args = args
        self.mixup = mixup
        self.dataset = args.dataset
        self.device = torch.device("cuda:%s" % args.gpu)

        if args.savename:
            self.SAVE_PATH = args.savename
        else:
            self.SAVE_PATH = rf"{dh.MODEL_DIR}/{self.dataset}_{DATETIME}"
        if exists(self.SAVE_PATH):
            shutil.rmtree(self.SAVE_PATH)
        os.makedirs(self.SAVE_PATH, exist_ok=True)

        with open(join(self.SAVE_PATH, "cfg.json"), "w") as outfile:
            json.dump(vars(args), outfile)

        scData_Train, self.scData_TrainLoader = self._scDataloader(train_and_test=self.args.train_and_test)
        scData_Test, self.scData_TestLoader = self._scDataloader(training=False)
        self.n_Train, self.n_Test = len(scData_Train), len(scData_Test)
        self.cell2cat = scData_Train.CELL2CAT
        self.n_vars = scData_Train.n_vars
        print("# Genes: %d" % self.n_vars)
        print("# Cells: %d (Training), %d (Testing)\n" % (self.n_Train * 256, self.n_Test * 256))

        self.MODEL = self._scModel()
        self._scTrain()

    def _scModel(self):
        if self.args.batch1d == "Vanilla":
            bn_eps, bn_momentum = 1e-5, 0.1
        elif self.args.batch1d == "scVI":
            bn_eps, bn_momentum = 1e-3, 0.01
        else:
            raise ValueError("Unknown batch1d type")

        return Model_ZOO[self.args.type](
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
            leak_dim=self.args.leakage,
            batchnorm=self.args.batchnorm,
            dropout_layer=self.args.dropout,
            mlp_size=self.args.mlp_size,
            cell2cat=self.cell2cat,
            n_gene=self.n_vars,
        ).to(self.device)

    def _scDataloader(self, training=True, train_and_test=False):
        _verb = True if training else False
        _scDataset = scDataset(
            dataset=self.dataset,
            verbose=_verb,
            training=training,
            train_and_test=train_and_test,
        )
        _scDataLoader = DataLoader(_scDataset, batch_size=1, shuffle=True, num_workers=8, collate_fn=collate_fn)
        return _scDataset, _scDataLoader

    def _loss_cet(self, emb_b, cell_type):
        emb_ = self.MODEL.projector(emb_b)
        logits_ = emb_.log_softmax(dim=-1)
        return F.nll_loss(logits_, cell_type.to(self.device))

    def _loss_mixup(self, emb_b, cell_type, alpha=1.0):
        # alpha = 1.0, 0.2, 0.4, 0.6

        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

        batch_size = emb_b.size()[0]
        index = torch.randperm(batch_size)
        y_a, y_b = cell_type, cell_type[index]
        mixed_x = lam * emb_b + (1 - lam) * emb_b[index, :]

        return lam * self._loss_cet(mixed_x, y_a) + (1 - lam) * self._loss_cet(mixed_x, y_b)

    def _loss_rec(self, counts_, rec_):
        return F.mse_loss(rec_, counts_)

    def _scTrain(self):
        LR = self.args.lr
        NUM_EPOCHS = self.args.epoch

        if self.args.optimiser == "adam":
            Opt = _Optimizer["adam"](self.MODEL.parameters(), lr=LR)
        elif args.optimiser == "sgd":
            Opt = _Optimizer["sgd"](self.MODEL.parameters(), lr=LR, momentum=0.9)
        else:
            raise ValueError("we now only work with adam and sgd optimisers")
        LR_SCHEDULER = CosineAnnealingLR(Opt, T_max=NUM_EPOCHS)

        LOCAL_PATIENCE = 0
        start = time.time()
        BEST_TEST_All = 1e4
        # BEST_TEST_Rec = 1e4
        # BEST_TEST_Celltype = 0
        for epoch in range(NUM_EPOCHS):
            """=== Training ==="""
            total_err, total_rec, total_cet = 0, 0, 0
            for _, batch_ in enumerate(self.scData_TrainLoader):
                Opt.zero_grad()

                counts_ = batch_["counts"].squeeze().to(self.device)
                rec_ = self.MODEL(counts_)
                emb_ = self.MODEL.extra_repr(counts_)

                loss_rec = self._loss_rec(counts_, rec_)
                loss_cet = self._loss_cet(emb_, batch_["cell_type"])
                if self.mixup:
                    loss_cet += self._loss_mixup(emb_, batch_["cell_type"])

                loss = self.args.w_rec * loss_rec + self.args.w_cet * loss_cet
                loss.backward()
                total_err += loss.item()
                total_rec += loss_rec.item()
                total_cet += loss_cet.item()
                Opt.step()
            LR_SCHEDULER.step()

            train_err = total_err / self.n_Train
            train_rec = total_rec / self.n_Train

            """ === Testing === """
            total_err, total_rec, total_acc = 0, 0, []
            for _, batch_ in enumerate(self.scData_TestLoader):
                with torch.no_grad():
                    counts_ = batch_["counts"].squeeze().to(self.device)
                    rec_ = self.MODEL(counts_)
                    emb_ = self.MODEL.extra_repr(counts_)

                    loss_rec = self._loss_rec(counts_, rec_)
                    loss_cet = self._loss_cet(emb_, batch_["cell_type"])
                    loss = self.args.w_rec * loss_rec + self.args.w_cet * loss_cet

                    total_err += loss.item()
                    total_rec += loss_rec.item()
                    total_acc.append(self.MODEL.acc_celltype_(emb_, batch_["cell_type"]))

            test_err = total_err / self.n_Test
            test_rec = total_rec / self.n_Test

            """ === Logging === """
            lr = LR_SCHEDULER.get_last_lr()[0]
            if epoch % 1 == 0:
                print(
                    "LR: %.6f " % lr,
                    "Epoch: %2d, " % epoch,
                    "Test Loss: %.2f, " % test_err,
                    "Train Loss: %.2f, " % train_err,
                    "Test Acc: %.2f, " % (100 * mean_(total_acc)),
                    "Time: %.2f Mins" % ((time.time() - start) // 60),
                )

            wandb.log(
                {
                    "LR": lr,
                    "Test Loss": test_err,
                    "Train Loss": train_err,
                    "Test Acc": 100 * mean_(total_acc),
                    #
                    "Train Rec Loss": train_rec,
                    "Test Rec Loss": test_rec,
                }
            )
            LOCAL_PATIENCE += 1

            if (epoch + 1) % 10 == 0:
                torch.save(self.MODEL.state_dict(), join(self.SAVE_PATH, "ckpt_%d.pth" % (epoch + 1)))

            if test_err < BEST_TEST_All:
                LOCAL_PATIENCE = 0
                BEST_TEST_All = test_err
                torch.save(self.MODEL.state_dict(), join(self.SAVE_PATH, "ckpt_best.pth"))

            if _isnan(train_err):
                print("NaN Value Encountered, Quitting")
                quit()

            if LOCAL_PATIENCE > PATIENCE:
                torch.save(self.MODEL.state_dict(), join(self.SAVE_PATH, "ckpt_%d.pth" % (epoch + 1)))
                print("Patience (%d Epochs) Reached, Quitting" % PATIENCE)
                quit()


class scTrainer_Semi(scTrainer):
    """Supervised Contrastive Learning"""

    def __init__(self, args, mode="scl"):
        self.mode = mode
        super().__init__(args=args)

    def _scDataloader(self, training=True, train_and_test=False):
        _verb = True if training else False
        _scDataset = scDataset(
            dataset=self.dataset,
            verbose=_verb,
            training=training,
            train_and_test=train_and_test,
        )
        if self.mode == "scl":
            _scDataset = ContrastiveDatasetDUAL(_scDataset)

        elif self.mode == "triplet":
            _scDataset = TripletDatasetDUAL(_scDataset)

        _scDataLoader = DataLoader(_scDataset, batch_size=1, shuffle=True, num_workers=8, collate_fn=collate_fn)
        return _scDataset, _scDataLoader

    def _scTrain(self):
        LR = self.args.lr
        NUM_EPOCHS = self.args.epoch

        if self.args.optimiser == "adam":
            Opt = _Optimizer["adam"](self.MODEL.parameters(), lr=LR)
        elif args.optimiser == "sgd":
            Opt = _Optimizer["sgd"](self.MODEL.parameters(), lr=LR, momentum=0.9)
        else:
            raise ValueError("we now only work with adam and sgd optimisers")

        Opt = Adam(self.MODEL.parameters(), lr=LR)
        LR_SCHEDULER = CosineAnnealingLR(Opt, T_max=NUM_EPOCHS)

        BEST_TEST = 1e4
        LOCAL_PATIENCE = 0
        start = time.time()

        triplet_loss = TripletLoss()
        scl_loss = SupervisedContrastiveLoss()

        for epoch in range(NUM_EPOCHS):
            train_err = 0
            if self.mode == "triplet":
                for item in self.scData_TrainLoader:
                    Opt.zero_grad()

                    anchor, positive, negative = item.values()
                    counts_a = anchor.to(self.device)
                    emb_a = self.MODEL.extra_repr(counts_a)

                    counts_p = positive.to(self.device)
                    emb_p = self.MODEL.extra_repr(counts_p)

                    counts_n = negative.to(self.device)
                    emb_n = self.MODEL.extra_repr(counts_n)

                    _loss = triplet_loss(emb_a, emb_p, emb_n)
                    _loss.backward()
                    Opt.step()
                    train_err += _loss.item()

            elif self.mode == "scl":
                for item in self.scData_TrainLoader:
                    b1, l1, b2, l2 = item.values()
                    Opt.zero_grad()

                    """ Intra Batch """
                    counts_1 = b1.squeeze().to(self.device)
                    emb_1 = self.MODEL.extra_repr(counts_1)
                    l1 = l1.to(self.device)

                    _permute = torch.randperm(emb_1.size()[0])
                    emb_, l1_ = emb_1[_permute], l1[_permute]
                    _loss = scl_loss(emb_1, emb_, l1, l1_)

                    """ Inter Batch """
                    counts_2 = b2.squeeze().to(self.device)
                    emb_2 = self.MODEL.extra_repr(counts_2)
                    if len(emb_1) == len(emb_2):
                        l2 = l2.to(self.device)
                        _permute = torch.randperm(emb_1.size()[0])
                        emb_2, l2 = emb_2[_permute], l2[_permute]
                        _loss += scl_loss(emb_1, emb_2, l1, l2)

                    _loss.backward()
                    Opt.step()
                    train_err += _loss.item()

            """ === Shared === """
            LR_SCHEDULER.step()
            lr = LR_SCHEDULER.get_last_lr()[0]
            train_err = train_err / len(self.scData_TrainLoader)

            if _isnan(train_err):
                self._str_formatter("NaN Value Encountered, Quitting")
                quit()

            if epoch % 1 == 0:
                print(
                    "LR: %.6f " % lr,
                    "Epoch: %2d, " % epoch,
                    "Total Loss: %.2f, " % train_err,
                    "EST: %.1f Mins" % ((time.time() - start) // 60),
                )

            if (epoch + 1) % 10 == 0:
                torch.save(self.MODEL.state_dict(), join(self.SAVE_PATH, "ckpt_%d.pth" % (epoch + 1)))

            LOCAL_PATIENCE += 1
            if train_err < BEST_TEST:
                LOCAL_PATIENCE = 0
                BEST_TEST = train_err
                torch.save(self.MODEL.state_dict(), join(self.SAVE_PATH, "ckpt_best.pth"))

            if LOCAL_PATIENCE > PATIENCE:
                torch.save(self.MODEL.state_dict(), join(self.SAVE_PATH, "ckpt_%d.pth" % (epoch + 1)))
                print("Patience (%d Epochs) Reached, Quitting" % PATIENCE)
                quit()

        return


class scTrainer_PSL(scTrainer):
    def __init__(self, args) -> None:
        pass


if __name__ == "__main__":
    args = Parser_Trainer()
    wandb.init(project=args.project, name=args.runname, config=args)

    if args.mode in ["vanilla", "mixup"]:
        scTrainer(args, mixup=args.mode == "mixup")
    elif args.mode in ["triplet", "scl"]:
        scTrainer_Semi(args, mode=args.mode)
    elif args.mode == "psl":
        scTrainer_PSL(args)
    else:
        raise ValueError("Unknown mode")
