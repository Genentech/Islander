from __future__ import absolute_import, division, print_function
import os, time, json, torch, wandb, shutil, Data_Handler as dh, numpy as np, torch.nn.functional as F
from torch.optim import SGD, RMSprop, Adam, ASGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from Utils_Handler import _isnan, mean_, DATETIME
from scDataset import scDataset, collate_fn
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

        scData_Train, self.scData_TrainLoader = self._scDataloader(
            train_and_test=self.args.train_and_test
        )
        scData_Test, self.scData_TestLoader = self._scDataloader(training=False)
        self.n_Train, self.n_Test = len(scData_Train), len(scData_Test)
        self.cell2cat = scData_Train.CELL2CAT
        self.n_vars = scData_Train.n_vars
        print("# Genes: %d" % self.n_vars)
        print(
            "# Cells: %d (Training), %d (Testing)\n"
            % (self.n_Train * 256, self.n_Test * 256)
        )

        # dict_ = vars(self.args)
        # dict_.update(self.cfg)
        # self.args = argparse.Namespace(**dict_)

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
        _scDataLoader = DataLoader(
            _scDataset, batch_size=1, shuffle=True, num_workers=8, collate_fn=collate_fn
        )
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

        return lam * self._loss_cet(mixed_x, y_a) + (1 - lam) * self._loss_cet(
            mixed_x, y_b
        )

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

        # TODO: try without scheduler
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
                    total_acc.append(
                        self.MODEL.acc_celltype_(emb_, batch_["cell_type"])
                    )

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
                torch.save(
                    self.MODEL.state_dict(),
                    join(self.SAVE_PATH, "ckpt_%d.pth" % (epoch + 1)),
                )

            if test_err < BEST_TEST_All:
                LOCAL_PATIENCE = 0
                BEST_TEST_All = test_err
                torch.save(
                    self.MODEL.state_dict(), join(self.SAVE_PATH, "ckpt_best.pth")
                )

            if _isnan(train_err):
                print("NaN Value Encountered, Quitting")
                quit()

            if LOCAL_PATIENCE > PATIENCE:
                torch.save(
                    self.MODEL.state_dict(),
                    join(self.SAVE_PATH, "ckpt_%d.pth" % (epoch + 1)),
                )
                print("Patience (%d Epochs) Reached, Quitting" % PATIENCE)
                quit()


class scTrainer_SCL(scTrainer):
    def __init__(self, args) -> None:
        pass


class scTrainer_PSL(scTrainer):
    def __init__(self, args) -> None:
        pass


if __name__ == "__main__":
    args = Parser_Trainer()
    wandb.init(project=args.project, name=args.runname, config=args)

    if args.mode in ["vanilla", "mixup"]:
        scTrainer(args, mixup=args.mode == "mixup")
    elif args.mode == "scl":
        scTrainer_SCL(args)
    elif args.mode == "psl":
        scTrainer_PSL(args)

    # DATASET = args.dataset
    # DEVICE = torch.device("cuda:%s" % args.gpu)
    # scData_Train = scDataset(dataset=DATASET, training=True)
    # scData_Test = scDataset(dataset=DATASET, verbose=False, training=False)

    # N_Train, N_Test = len(scData_Train), len(scData_Test)

    # if args.savename:
    #     SAVE_PATH = args.savename
    # else:
    #     SAVE_PATH = rf"{dh.MODEL_DIR}/{DATASET}_{DATETIME}"
    # if exists(SAVE_PATH):
    #     shutil.rmtree(SAVE_PATH)
    # os.makedirs(SAVE_PATH, exist_ok=True)

    # with open(join(SAVE_PATH, "cfg.json"), "w") as outfile:
    #     json.dump(vars(args), outfile)

    # scData_TrainLoader = DataLoader(
    #     scData_Train, batch_size=1, shuffle=True, num_workers=8, collate_fn=collate_fn
    # )
    # scData_TestLoader = DataLoader(
    #     scData_Test, batch_size=1, shuffle=True, num_workers=8, collate_fn=collate_fn
    # )

    # print("# of Genes: %d" % scData_Train.n_vars)
    # print(
    #     "# of Cells: %d (Training), %d (Testing)\n"
    #     % (scData_Train.n_obs, scData_Test.n_obs)
    # )

    # if args.batch1d == "Vanilla":
    #     bn_eps, bn_momentum = 1e-5, 0.1
    # elif args.batch1d == "scVI":
    #     bn_eps, bn_momentum = 1e-3, 0.01
    # else:
    #     raise ValueError("Unknown batch1d type")

    # MODEL = Model_ZOO[args.type](
    #     bn_eps=bn_eps,
    #     mlp_size=args.mlp_size,
    #     bn_momentum=bn_momentum,
    #     batchnorm=args.batchnorm,
    #     dropout_layer=args.dropout,
    #     n_gene=scData_Train.n_vars,
    # ).to(device=DEVICE)

    # LR = args.lr
    # COUNTS = args.counts
    # NUM_EPOCHS = args.epoch

    # if args.optimiser == "adam":
    #     Opt = _Optimizer[args.optimiser](MODEL.parameters(), lr=LR)
    # elif args.optimiser == "sgd":
    #     Opt = _Optimizer[args.optimiser](MODEL.parameters(), lr=LR, momentum=0.9)
    # else:
    #     raise ValueError("we now only work with adam and sgd optimisers")

    # assert args.lr_scheduler == "cosine"
    # LR_SCHEDULER = CosineAnnealingLR(Opt, T_max=NUM_EPOCHS)

    # if args.load_from_save:
    #     assert args.saved_checkpoint, "the saved checkpoint should be provided"
    #     MODEL.load_state_dict(torch.load(args.saved_checkpoint))

    # BEST_TEST_All = 1e4
    # BEST_TEST_Rec = 1e4
    # BEST_TEST_Celltype = 0

    # LOCAL_PATIENCE = 0
    # start = time.time()
    # for epoch in range(NUM_EPOCHS):
    #     """=== Training ==="""
    #     total_, total_rec, total_celltype = 0, 0, 0
    #     for _, batch_ in enumerate(scData_TrainLoader):
    #         Opt.zero_grad()
    #         counts_ = batch_[COUNTS].squeeze().to(DEVICE)
    #         rec_ = MODEL(counts_)
    #         emb_ = MODEL.extra_repr(counts_)

    #         loss_rec = MODEL._loss_rec(counts_, rec_)
    #         loss_celltype = MODEL._loss_celltype(emb_, batch_["cell_type"])

    #         loss = loss_rec + args.w_cet * loss_celltype

    #         loss.backward()
    #         total_ += loss.item()
    #         total_rec += loss_rec.item()
    #         total_celltype += loss_celltype.item()
    #         Opt.step()
    #     LR_SCHEDULER.step()
    #     train_loss = total_ / N_Train
    #     train_rec_loss = total_rec / N_Train

    #     """ === Testing === """
    #     total_, total_rec, _acc = 0, 0, []
    #     for _, batch_ in enumerate(scData_TestLoader):
    #         with torch.no_grad():
    #             counts_ = batch_[COUNTS].squeeze().to(DEVICE)
    #             rec_ = MODEL(counts_)

    #             emb_ = MODEL.extra_repr(counts_)

    #             loss_rec = MODEL._loss_rec(counts_, rec_)
    #             loss_celltype = args.w_cet * MODEL._loss_celltype(
    #                 emb_, batch_["cell_type"]
    #             )

    #             loss = loss_rec + loss_celltype

    #             total_ += loss.item()
    #             total_rec += loss_rec.item()
    #             # total_celltype += loss_celltype.item()
    #             _acc.append(MODEL.acc_celltype_(emb_, batch_["cell_type"]))

    #     test_loss = total_ / N_Test
    #     test_rec_loss = total_rec / N_Test

    #     """ === Logging === """
    #     lr = LR_SCHEDULER.get_last_lr()[0]
    #     if epoch % 1 == 0:
    #         print(
    #             "LR: %.3f " % lr,
    #             "Epoch: %2d, " % epoch,
    #             "Test Loss: %.2f, " % test_loss,
    #             "Train Loss: %.2f, " % train_loss,
    #             "Test Acc: %.2f, " % (100 * mean_(_acc)),
    #             "Time: %.2f Mins" % ((time.time() - start) // 60),
    #         )

    #     wandb.log(
    #         {
    #             "LR": lr,
    #             "Test Loss": test_loss,
    #             "Train Loss": train_loss,
    #             "Test Acc": 100 * mean_(_acc),
    #             #
    #             "Train Rec Loss": train_rec_loss,
    #             "Test Rec Loss": test_rec_loss,
    #         }
    #     )
    #     LOCAL_PATIENCE += 1

    #     if (epoch + 1) % 10 == 0:
    #         torch.save(MODEL.state_dict(), join(SAVE_PATH, "ckpt_%d.pth" % (epoch + 1)))

    #     if test_loss < BEST_TEST_All:
    #         LOCAL_PATIENCE = 0
    #         BEST_TEST_All = test_loss
    #         torch.save(MODEL.state_dict(), join(SAVE_PATH, "ckpt_best.pth"))

    #     # if test_rec_loss < BEST_TEST_Rec and args.w_rec > 0:
    #     #     LOCAL_PATIENCE = 0
    #     #     BEST_TEST_Reconstruct = test_rec_loss
    #     #     torch.save(MODEL.state_dict(), join(SAVE_PATH, "ckpt_best_reconstruct.pth"))

    #     # if mean_(cell_acc_) > BEST_TEST_Celltype and args.w_cet > 0:
    #     #     LOCAL_PATIENCE = 0
    #     #     BEST_TEST_Celltype = mean_(cell_acc_)
    #     #     torch.save(MODEL.state_dict(), join(SAVE_PATH, "ckpt_best_celltype.pth"))

    #     if _isnan(train_loss):
    #         print("NaN Value Encountered, Quitting")
    #         quit()

    #     if LOCAL_PATIENCE > PATIENCE:
    #         torch.save(MODEL.state_dict(), join(SAVE_PATH, "ckpt_%d.pth" % (epoch + 1)))
    #         print("Patience (%d Epochs) Reached, Quitting" % PATIENCE)
    #         quit()
