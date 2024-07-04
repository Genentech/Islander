# https://docs.scvi-tools.org/en/stable/tutorials/notebooks/harmonization.html
# https://scib-metrics.readthedocs.io/en/stable/notebooks/lung_example.html

import Data_Handler as dh
import os, scvi, json, torch, argparse, numpy as np, scanpy as sc, pandas as pd
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
from scDataset import scDataset, collate_fn, set_seed
from os.path import join, exists, getmtime, dirname
from ArgParser import Parser_Benchmarker
from torch.utils.data import DataLoader
from scModel import Model_ZOO

# NOTE: We downgrade the version of JAXLAB and JAX
# https://github.com/google/jax/issues/15268#issuecomment-1487625083


set_seed(dh.SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = dh.CUDA_DEVICE
# torch.cuda.is_available() -> False
sc.settings.verbosity = 3
sc.logging.print_header()

GPU_4_NEIGHBORS = False

print("\n\n")
print("=" * 44)
print("use GPU for neighbors calculation: ", GPU_4_NEIGHBORS)
print("GPU is available: ", torch.cuda.is_available())
print("=" * 44)
print("\n")


def faiss_hnsw_nn(X: np.ndarray, k: int):
    # """GPU HNSW nearest neighbor search using faiss.

    # See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
    # for index param details.
    # """
    # X = np.ascontiguousarray(X, dtype=np.float32)
    # res = faiss.StandardGpuResources()
    # M = 32
    # index = faiss.IndexHNSWFlat(X.shape[1], M, faiss.METRIC_L2)
    # gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    # gpu_index.add(X)
    # distances, indices = gpu_index.search(X, k)
    # del index
    # del gpu_index
    # # distances are squared
    # return NeighborsOutput(indices=indices, distances=np.sqrt(distances))
    raise NotImplementedError


def faiss_brute_force_nn(X: np.ndarray, k: int):
    # """GPU brute force nearest neighbor search using faiss."""
    # X = np.ascontiguousarray(X, dtype=np.float32)
    # res = faiss.StandardGpuResources()
    # index = faiss.IndexFlatL2(X.shape[1])
    # gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    # gpu_index.add(X)
    # distances, indices = gpu_index.search(X, k)
    # del index
    # del gpu_index
    # # distances are squared
    # return NeighborsOutput(indices=indices, distances=np.sqrt(distances))
    raise NotImplementedError


class BasicHandler:
    def __init__(self, args, ext_adata=None):
        self.args = args
        self.parse_cfg()
        self.ext_adata = ext_adata
        self.device = torch.device("cuda:0")
        self.res_dir = rf"{dh.RES_DIR}/scCB"
        os.makedirs(self.res_dir, exist_ok=True)

    def _str_formatter(self, message):
        print(f"\n=== {message} ===\n")

    def _scDataloader(self, shuffle=True):
        # assert exists(join(dh.DATA_DIR, self.data_prefix)), "datapath should exist"

        if self.ext_adata:
            self.adata = self.ext_adata
            self.external_adata = True
            self._str_formatter("Using External ADATA")

        else:  # HLCA
            self.adata = dh.DATA_RAW

        _batch_id = list(dh.BATCH2CAT.keys())
        if self.args.batch_id:
            _batch_id = self.args.batch_id

        _dataset = scDataset(
            self.adata,
            inference=True,
            batch_id=_batch_id,
            prefix=self.data_prefix,
            n_cells=self.args.n_cells,
        )
        self.n_gene = _dataset.n_vars

        if len(_batch_id) == 1:
            # Generating data counteraparts based on the given batches
            self._str_formatter("Single Batch: {}".format(_batch_id))
            batch_ = _dataset[0]
            _df = pd.DataFrame(index=dh.CONCEPTS2CAT.keys(), columns=["digits", "names"])
            _df["digits"] = np.concatenate(list(batch_["meta"].values())).reshape(dh.NUM_CONCEPTS, -1).astype(int)[:, 0]
            _df["names"] = dh.CONCEPT2NAMES(_df["digits"])
            print(_df)
        else:
            self._str_formatter("Multi Batches: {}".format(_batch_id))

        return DataLoader(
            _dataset,
            batch_size=1,
            num_workers=4,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

    def _load_model(self):
        if self.args.batch1d == "Vanilla":
            bn_eps, bn_momentum = 1e-5, 0.1
        elif self.args.batch1d == "scVI":
            bn_eps, bn_momentum = 1e-3, 0.01
        else:
            raise ValueError("Unknown batch1d type")

        self.model = Model_ZOO[self.args.type](
            n_gene=self.n_gene,
            leak_dim=self.args.leakage,
            mlp_size=self.args.mlp_size,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
            batchnorm=self.args.batchnorm,
            dropout_layer=self.args.dropout,
            with_projector=False,
        ).to(self.device)

        self._load_part_of_state_dict(torch.load(self.ckpt_path))
        self._str_formatter("Loaded weights from: %s" % self.ckpt_path)
        self.model.eval()

    def _load_part_of_state_dict(self, state_dict):
        model_state_dict = self.model.state_dict()
        common_keys = set(model_state_dict.keys()) & set(state_dict.keys())
        extra_keys = set(state_dict.keys()) - set(model_state_dict.keys())
        missing_keys = set(model_state_dict.keys()) - set(state_dict.keys())

        for key in common_keys:
            model_state_dict[key] = state_dict[key]

        self.model.load_state_dict(model_state_dict)

        if extra_keys:
            print(f"Warning: The following keys present in the checkpoints are not found in the model: {extra_keys}")
        if missing_keys:
            print(f"Warning: The following keys present in the model are not found in the checkpoints: {missing_keys}")

    def parse_cfg(self):
        def _recent_ckpt(dirpath):
            a = [s for s in os.listdir(dirpath) if ".pth" in s]
            a.sort(key=lambda s: getmtime(join(dirpath, s)))
            return a

        self.cfg = None
        if ".pth" in self.args.save_path:
            self.ckpt_path = self.args.save_path
        else:
            if exists(join(self.args.save_path, "ckpt_best.pth")):
                self.ckpt_path = join(self.args.save_path, "ckpt_best.pth")
            else:
                self.ckpt_path = join(self.args.save_path, _recent_ckpt(self.args.save_path)[-1])
        if exists(join(self.args.save_path, "cfg.json")):
            self.cfg = json.load(open(join(self.args.save_path, "cfg.json")))
        elif exists(join(dirname(self.args.save_path), "cfg.json")):
            self.cfg = json.load(open(join(dirname(self.args.save_path), "cfg.json")))

        if self.cfg:
            dict_ = vars(self.args)
            dict_.update(self.cfg)
            self.args = argparse.Namespace(**dict_)

        assert ("concept" in self.args.type) or ("cb" in self.args.type)


class scIB(BasicHandler):
    """Integration Benchmarker"""

    # https://scib-metrics.readthedocs.io/en/stable/
    # https://github.com/theislab/scib-reproducibility/
    # https://github.com/theislab/scib-pipeline/
    # https://github.com/theislab/scib/

    def __init__(self, args):
        super().__init__(args=args)
        self._load_adata_()

    def _load_adata_(self):
        _suffix = "_hvg" if self.args.highvar else ""
        assert not self.args.use_raw, "use_raw is not supported"
        self.adata = sc.read(dh.DATA_EMB_[self.args.dataset + _suffix])
        #
        self.batch_key = dh.META_[self.args.dataset]["batch"]
        self.label_key = dh.META_[self.args.dataset]["celltype"]
        return

    def _str_formatter(self, message):
        print(f"\n=== {message} ===\n")

    def _scDataloader(self):
        self._str_formatter("Dataloader")
        _verb = True
        _scDataset = scDataset(
            dataset=self.args.dataset,
            inference=True,
            rm_cache=False,
            verbose=_verb,
        )
        _scDataLoader = DataLoader(
            _scDataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn,
        )
        return _scDataset, _scDataLoader

    def _benchmark_(self, n_jobs=-1, scratch=False):
        # NOTE: for DEBUG
        # first_5_ = self.adata.obs[self.batch_key].unique()[:5].to_list()
        # self.adata = self.adata[[_it in first_5_ for _it in self.adata.obs[self.batch_key]]]

        # recompute the embeddings
        # self.scratch = True if self.args.highvar else scratch
        self.scratch = scratch
        self._pca_()
        if self.args.umap or self.args.all:
            self._umap_()
        if self.args.tsne or self.args.all:
            self._tsne_()
        if self.args.harmony or self.args.all:
            self._harmony_()
        if self.args.scanorama or self.args.all:
            self._scanorama_()
        if self.args.scvi or self.args.all:
            self._scvi_()
        if self.args.scanvi or self.args.all:
            self._scanvi_()
        if self.args.bbknn or self.args.all:
            self._bbknn_()
        if self.args.scgen or self.args.all:
            self._scgen_()
        if self.args.fastmnn or self.args.all:
            self._fastmnn_()
        if self.args.scpoli or self.args.all:
            self._scpoli_()

        if self.args.islander:
            self._islander_()

        if self.args.obsm_keys is None:
            obsm_keys = list(self.adata.obsm)
        else:
            obsm_keys = self.args.obsm_keys

        for embed in obsm_keys:
            print("%12s, %d" % (embed, self.adata.obsm[embed].shape[1]))
            if self.adata.obsm[embed].shape[0] != np.unique(self.adata.obsm[embed], axis=0).shape[0]:
                print("\nWarning: Embedding %s has duplications\n" % embed)
                obsm_keys.remove(embed)
        if self.args.saveadata or self.args.all:
            _suffix = "_hvg" if self.args.highvar else ""
            self.adata.write_h5ad(dh.DATA_EMB_[self.args.dataset + _suffix], compression="gzip")

        self._str_formatter(rf"scIB Benchmarking: {obsm_keys}")
        biocons = BioConservation(nmi_ari_cluster_labels_leiden=True)
        batcorr = BatchCorrection()

        """ === for DEBUG ==="""
        # biocons = BioConservation(isolated_labels=False)
        # biocons = BioConservation(
        #     silhouette_label=False,
        #     isolated_labels=False,)
        # batcorr = BatchCorrection(
        #     silhouette_batch=False,
        #     ilisi_knn=False,
        #     kbet_per_label=False,
        # )

        self.benchmarker = Benchmarker(
            self.adata,
            batch_key=self.batch_key,
            label_key=self.label_key,
            embedding_obsm_keys=obsm_keys,
            bio_conservation_metrics=biocons,
            batch_correction_metrics=batcorr,
            pre_integrated_embedding_obsm_key="X_pca",
            n_jobs=n_jobs,
        )
        if torch.cuda.is_available() and GPU_4_NEIGHBORS:
            # self.benchmarker.prepare(neighbor_computer=faiss_brute_force_nn)
            self.benchmarker.prepare(neighbor_computer=faiss_hnsw_nn)
        else:
            # Calculate the Neighbors based on the CPUs
            self.benchmarker.prepare(neighbor_computer=None)
        self.benchmarker.benchmark()

        self._str_formatter("scIB Benchmarking Finished")
        df = self.benchmarker.get_results(min_max_scale=False)
        print(df.head(7))

        os.makedirs(rf"{dh.RES_DIR}/scIB", exist_ok=True)
        if self.args.savecsv is not None:
            df.to_csv(rf"{dh.RES_DIR}/scIB/{self.args.savecsv}.csv")
        else:
            savecsv = self.args.save_path.split("/")[-1].replace(" ", "_")
            df.to_csv(rf"{dh.RES_DIR}/scIB/{savecsv}.csv")

        savefig = False
        if savefig:
            _suffix = "_hvg" if self.args.highvar else ""
            self.benchmarker.plot_results_table(min_max_scale=False, show=False, save_dir=rf"{dh.RES_DIR}/scIB/")
            os.rename(
                src=rf"{dh.RES_DIR}/scIB/scib_results.svg",
                dst=rf"{dh.RES_DIR}/figures/scib_{self.args.dataset}{_suffix}.svg",
            )
        return

    def _save_adata_(self):
        if self.args.saveadata or self.args.all:
            _suffix = "_hvg" if self.args.highvar else ""
            self._str_formatter("Saving %s" % (self.args.dataset + _suffix))
            self.adata.write_h5ad(dh.DATA_EMB_[self.args.dataset + _suffix], compression="gzip")
        return

    def _pca_(self, n_comps=50):
        self._str_formatter("PCA")
        if "X_pca" in self.adata.obsm and not self.scratch:
            return
        sc.pp.pca(self.adata, n_comps=n_comps)
        self._save_adata_()
        return

    def _umap_(self, n_neighbors=10, n_pcs=50):
        self._str_formatter("UMAP")
        if "X_umap" in self.adata.obsm and not self.scratch:
            return
        sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.umap(self.adata)
        self._save_adata_()
        return

    def _tsne_(self):
        if "X_tsne" in self.adata.obsm and not self.scratch:
            return
        self._str_formatter("TSNE")
        sc.tl.tsne(self.adata)
        self._save_adata_()
        return

    def _scvi_(self):
        self._str_formatter("scVI")
        if "X_scVI" in self.adata.obsm and not self.scratch:
            return
        adata = self.adata.copy()
        # adata.X = adata.layers["raw_counts"]
        scvi.model.SCVI.setup_anndata(adata, layer=None, batch_key=self.batch_key)
        self.vae = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
        self.vae.train()
        self.adata.obsm["X_scVI"] = self.vae.get_latent_representation()
        self._save_adata_()
        return

    def _scanvi_(self):
        self._str_formatter("X_scANVI")
        if "X_scANVI" in self.adata.obsm and not self.scratch:
            return
        lvae = scvi.model.SCANVI.from_scvi_model(
            self.vae,
            adata=self.adata,
            labels_key=self.label_key,
            unlabeled_category="Unknown",
        )
        lvae.train()
        self.adata.obsm["X_scANVI"] = lvae.get_latent_representation(self.adata)
        # self.adata.obsm["mde_scanvi"] = mde(self.adata.obsm["X_scANVI"])
        self._save_adata_()
        return

    def _bbknn_(self):
        # ref: https://bbknn.readthedocs.io/en/latest/#
        # tutorial: https://nbviewer.org/github/Teichlab/bbknn/blob/master/examples/mouse.ipynb

        import bbknn

        self._str_formatter("bbknn")
        if "X_bbknn" in self.adata.obsm and not self.scratch:
            return
        # if self.adata.n_obs < 1e5:
        #     _temp_adata = bbknn.bbknn(self.adata, batch_key=self.batch_key, copy=True)
        # else:
        print(self.adata.obs[self.batch_key].value_counts().tail())
        _smallest_n_neighbor = self.adata.obs[self.batch_key].value_counts().tail(1).values[0]
        _temp_adata = bbknn.bbknn(
            self.adata,
            batch_key=self.batch_key,
            neighbors_within_batch=min(10, _smallest_n_neighbor),
            copy=True,
        )
        sc.tl.umap(_temp_adata)
        self.adata.obsm["X_bbknn"] = _temp_adata.obsm["X_umap"]
        self._save_adata_()
        return

    def _harmony_(self):
        # https://pypi.org/project/harmony-pytorch/
        self._str_formatter("Harmony")
        if "Harmony" in self.adata.obsm and not self.scratch:
            return

        from harmony import harmonize

        self.adata.obsm["Harmony"] = harmonize(self.adata.obsm["X_pca"], self.adata.obs, batch_key=self.batch_key)
        self._save_adata_()
        return

    def _scanorama_(self):
        # https://github.com/brianhie/scanorama
        self._str_formatter("Scanorama")
        if "Scanorama" in self.adata.obsm and not self.scratch:
            return

        import scanorama

        batch_cats = self.adata.obs[self.batch_key].cat.categories
        adata_list = [self.adata[self.adata.obs[self.batch_key] == b].copy() for b in batch_cats]
        scanorama.integrate_scanpy(adata_list)

        self.adata.obsm["Scanorama"] = np.zeros((self.adata.shape[0], adata_list[0].obsm["X_scanorama"].shape[1]))
        for i, b in enumerate(batch_cats):
            self.adata.obsm["Scanorama"][self.adata.obs[self.batch_key] == b] = adata_list[i].obsm["X_scanorama"]
        self._save_adata_()
        return

    def _scgen_(self):
        self._str_formatter("scGen")
        if "scgen_pca" in self.adata.obsm and not self.scratch:
            return

        if not self.args.highvar:
            return

        import scgen

        # ref: https://scgen.readthedocs.io/en/stable/tutorials/scgen_batch_removal.html
        # pip install git+https://github.com/theislab/scgen.git

        # ref: https://github.com/theislab/scib-reproducibility/blob/main/notebooks/integration/Test_scgen.ipynb
        # ref: https://github.com/theislab/scib/blob/main/scib/integration.py

        # ref: https://github.com/LungCellAtlas/HLCA_reproducibility/blob/main/notebooks/1_building_and_annotating_the_atlas_core/04_integration_benchmark_prep_and_scgen.ipynb

        scgen.SCGEN.setup_anndata(self.adata, batch_key=self.batch_key, labels_key=self.label_key)
        model = scgen.SCGEN(self.adata)
        model.train(
            max_epochs=100,
            batch_size=32,  # 32
            early_stopping=True,
            early_stopping_patience=25,
        )

        adata_scgen = model.batch_removal()
        sc.pp.pca(adata_scgen, svd_solver="arpack")
        sc.pp.neighbors(adata_scgen)
        sc.tl.umap(adata_scgen)

        self.adata.obsm["scgen_umap"] = adata_scgen.obsm["X_umap"]
        self.adata.obsm["scgen_pca"] = adata_scgen.obsm["X_pca"]
        self._save_adata_()
        return

    def _fastmnn_(self):
        # [deprecated]: https://github.com/chriscainx/mnnpy
        # ref: https://github.com/HelloWorldLTY/mnnpy
        # from: https://github.com/chriscainx/mnnpy/issues/42

        self._str_formatter("fastMNN")
        if "fastMNN_pca" in self.adata.obsm and not self.scratch:
            return

        if not self.args.highvar:
            return

        import mnnpy

        def split_batches(adata, batch_key, hvg=None, return_categories=False):
            """Split batches and preserve category information
            Ref: https://github.com/theislab/scib/blob/main/scib/utils.py#L32"""
            split = []
            batch_categories = adata.obs[batch_key].cat.categories
            if hvg is not None:
                adata = adata[:, hvg]
            for i in batch_categories:
                split.append(adata[adata.obs[batch_key] == i].copy())
            if return_categories:
                return split, batch_categories
            return split

        split, categories = split_batches(self.adata, batch_key=self.batch_key, return_categories=True)
        if self.args.dataset in [
            "lung_fetal_organoid",
            "COVID",
            "heart",
            "brain",
            "breast",
        ]:
            k = 10
        else:
            k = 20
        corrected, _, _ = mnnpy.mnn_correct(
            *split,
            k=k,
            batch_key=self.batch_key,
            batch_categories=categories,
            index_unique=None,
        )

        adata_fastmnn = corrected
        sc.pp.pca(adata_fastmnn, svd_solver="arpack")
        sc.pp.neighbors(adata_fastmnn)
        sc.tl.umap(adata_fastmnn)

        self.adata.obsm["fastMNN_umap"] = adata_fastmnn.obsm["X_umap"]
        self.adata.obsm["fastMNN_pca"] = adata_fastmnn.obsm["X_pca"]
        self._save_adata_()
        return

    def _scpoli_(self):
        self._str_formatter("scPoli")
        if "scPoli" in self.adata.obsm and not self.scratch:
            return
        if self.args.dataset in ["brain", "breast"]:
            return
        import warnings

        warnings.filterwarnings("ignore")
        from scarches.models.scpoli import scPoli

        self.adata.X = self.adata.X.astype(np.float32)
        early_stopping_kwargs = {
            "early_stopping_metric": "val_prototype_loss",
            "mode": "min",
            "threshold": 0,
            "patience": 10,
            "reduce_lr": True,
            "lr_patience": 13,
            "lr_factor": 0.1,
        }
        scpoli_model = scPoli(
            adata=self.adata,
            condition_keys=self.batch_key,
            cell_type_keys=self.label_key,
            embedding_dims=16,
            recon_loss="nb",
        )
        scpoli_model.train(
            n_epochs=50,
            pretraining_epochs=40,
            early_stopping_kwargs=early_stopping_kwargs,
            eta=5,
        )
        self.adata.obsm["scPoli"] = scpoli_model.get_latent(self.adata, mean=True)
        return

    def _islander_(self):
        from tqdm import tqdm

        scDataset, self.scDataLoader = self._scDataloader()
        # self.cell2cat = scData_Train.CELL2CAT
        self.n_gene = scDataset.n_vars
        self._load_model()
        self.model.eval()
        emb_cells = []

        for item in tqdm(self.scDataLoader):
            counts_ = item["counts"].to(self.device).squeeze()
            # for _idx in range(self.adata.n_obs):
            #     if (self.adata.X[_idx, :] == counts_.cpu().numpy()[1]).all():
            #         print(_idx)
            emb_cell = self.model.extra_repr(counts_)
            emb_cells.append(emb_cell.detach().cpu().numpy())

        emb_cells = np.concatenate(emb_cells, axis=0)
        self.adata.obsm["Islander"] = emb_cells
        return


if __name__ == "__main__":
    #
    args = Parser_Benchmarker()
    benchmarker = scIB(args=args)
    benchmarker._benchmark_()

    # jaxlib is version 0.4.7, but this version of jax requires version >= 0.4.14.
