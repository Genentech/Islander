# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import scanpy as sc
from geneformer import EmbExtractor

Dataset_Lists = ["lung_fetal_organoid", "lung_fetal_donor", "brain", "breast"]

for Dataset in Dataset_Lists:
    DATA_DIR = "/home/wangh256/G-scIB_dev/data/%s/" % Dataset
    embex = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        max_ncells=None,
        forward_batch_size=128,
        nproc=16,
    )

    embs = embex.extract_embs(
        "/home/wangh256/GeneDataEngine_dev/Geneformer-main/geneformer-12L-30M",
        rf"{DATA_DIR}/Geneformer/%s.dataset" % Dataset,
        rf"{DATA_DIR}/",
        "Geneformer",
    )

    bdata = sc.read_h5ad(rf"{DATA_DIR}/emb.h5ad")
    bdata.obsm["Geneformer"] = embs.values
    bdata.write_h5ad(rf"{DATA_DIR}/emb.h5ad", compression="gzip")
