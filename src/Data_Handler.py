import os, json, warnings
from anndata import ImplicitModificationWarning
from numba.core.errors import NumbaDeprecationWarning, NumbaWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NumbaWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=ImplicitModificationWarning)

SEED = 42
MINCELLS = 256
CUDA_DEVICE = "1"

HOME_DIR = os.path.expanduser("~")
PROJ_DIR = rf"{HOME_DIR}/G-scIB_dev"
MODEL_DIR = rf"{PROJ_DIR}/models"
META_DIR = rf"{PROJ_DIR}/meta"
DATA_DIR = rf"{PROJ_DIR}/data"
RES_DIR = rf"{PROJ_DIR}/res"

os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(rf"{RES_DIR}/scIB", exist_ok=True)

DATA_RAW_ = {
    #
    "lung": rf"{DATA_DIR}/lung/local.h5ad",
    #
    "lung_fetal_donor": rf"{DATA_DIR}/lung_fetal_donor/donor.h5ad",
    #
    "lung_fetal_organoid": rf"{DATA_DIR}/lung_fetal_organoid/organoid.h5ad",
    #
    "brain": rf"{DATA_DIR}/brain/local.h5ad",
    #
    "breast": rf"{DATA_DIR}/breast/local.h5ad",
    #
    "heart": rf"{DATA_DIR}/heart/local.h5ad",
    #
    "eye": rf"{DATA_DIR}/eye/local.h5ad",
    #
    "gut_fetal": rf"{DATA_DIR}/gut_fetal/local.h5ad",
    #
    "skin": rf"{DATA_DIR}/skin/local.h5ad",
    #
    "COVID": rf"{DATA_DIR}/COVID/local.h5ad",
    #
    "pancreas": rf"{DATA_DIR}/pancreas/human_pancreas_norm_complexBatch.h5ad",  # need to deal with this
}

DATA_EMB_ = {
    "lung": rf"{DATA_DIR}/lung/emb.h5ad",
    "lung_hvg": rf"{DATA_DIR}/lung/emb_hvg.h5ad",
    "lung_fetal_donor": rf"{PROJ_DIR}/data/lung_fetal_donor/emb.h5ad",
    "lung_fetal_donor_hvg": rf"{PROJ_DIR}/data/lung_fetal_donor/emb_hvg.h5ad",
    "lung_fetal_organoid": rf"{PROJ_DIR}/data/lung_fetal_organoid/emb.h5ad",
    "lung_fetal_organoid_hvg": rf"{PROJ_DIR}/data/lung_fetal_organoid/emb_hvg.h5ad",
    "brain": rf"{DATA_DIR}/brain/emb.h5ad",
    "brain_hvg": rf"{DATA_DIR}/brain/emb_hvg.h5ad",
    "breast": rf"{DATA_DIR}/breast/emb.h5ad",
    "breast_hvg": rf"{DATA_DIR}/breast/emb_hvg.h5ad",
    "heart": rf"{DATA_DIR}/heart/emb.h5ad",
    "heart_hvg": rf"{DATA_DIR}/heart/emb_hvg.h5ad",
    "eye": rf"{DATA_DIR}/eye/emb.h5ad",
    "eye_hvg": rf"{DATA_DIR}/eye/emb_hvg.h5ad",
    "gut_fetal": rf"{DATA_DIR}/gut_fetal/emb.h5ad",
    "gut_fetal_hvg": rf"{DATA_DIR}/gut_fetal/emb_hvg.h5ad",
    "skin": rf"{DATA_DIR}/skin/emb.h5ad",
    "skin_hvg": rf"{DATA_DIR}/skin/emb_hvg.h5ad",
    "COVID": rf"{DATA_DIR}/COVID/emb.h5ad",
    "COVID_hvg": rf"{DATA_DIR}/COVID/emb_hvg.h5ad",
    "pancreas": rf"{DATA_DIR}/pancreas/emb.h5ad",
    "pancreas_hvg": rf"{DATA_DIR}/pancreas/emb_hvg.h5ad",
}

META_ = {
    "lung": {"batch": "sample", "celltype": "cell_type"},
    "lung_fetal_donor": {"batch": "batch", "celltype": "new_celltype"},
    # "lung_fetal_donor": {"batch": "batch", "celltype": "broad_celltype"},
    "lung_fetal_organoid": {"batch": "batch", "celltype": "new_celltype"},
    "brain": {"batch": "donor_id", "celltype": "cell_type"},  # or "sample_id"
    "breast": {"batch": "donor_id", "celltype": "cell_type"},  # or "sample_id"
    "heart": {"batch": "donor_id", "celltype": "cell_type"},
    "eye": {"batch": "biosample_id", "celltype": "cell_type"},
    "gut_fetal": {"batch": "donor_id", "celltype": "cell_type"},  # or "Sample"
    "skin": {"batch": "donor_id", "celltype": "cell_type"},
    "COVID": {"batch": "batch", "celltype": "predicted.celltype.l2"},  # or "batch"
    "pancreas": {"batch": "tech", "celltype": "celltype"},
}

CELL2CAT_ = {
    "lung": json.loads(open(rf"{META_DIR}/lung/cell2cat.json", "r").read()),
    "lung_fetal_donor": json.loads(
        open(rf"{META_DIR}/lung_fetal_donor/cell2cat.json", "r").read()
    ),
    "lung_fetal_organoid": json.loads(
        open(rf"{META_DIR}/lung_fetal_organoid/cell2cat.json", "r").read()
    ),
    "brain": json.loads(open(rf"{META_DIR}/brain/cell2cat.json", "r").read()),
    "breast": json.loads(open(rf"{META_DIR}/breast/cell2cat.json", "r").read()),
    "heart": json.loads(open(rf"{META_DIR}/heart/cell2cat.json", "r").read()),
    "eye": json.loads(open(rf"{META_DIR}/eye/cell2cat.json", "r").read()),
    "gut_fetal": json.loads(open(rf"{META_DIR}/gut_fetal/cell2cat.json", "r").read()),
    "skin": json.loads(open(rf"{META_DIR}/skin/cell2cat.json", "r").read()),
    "COVID": json.loads(open(rf"{META_DIR}/COVID/cell2cat.json", "r").read()),
    "pancreas": json.loads(open(rf"{META_DIR}/pancreas/cell2cat.json", "r").read()),
}

BATCH2CAT_ = {
    "lung": json.loads(open(rf"{META_DIR}/lung/batch2cat.json", "r").read()),
    "lung_fetal_donor": json.loads(
        open(rf"{META_DIR}/lung_fetal_donor/batch2cat.json", "r").read()
    ),
    "lung_fetal_organoid": json.loads(
        open(rf"{META_DIR}/lung_fetal_organoid/batch2cat.json", "r").read()
    ),
    "brain": json.loads(open(rf"{META_DIR}/brain/batch2cat.json", "r").read()),
    "breast": json.loads(open(rf"{META_DIR}/breast/batch2cat.json", "r").read()),
    "heart": json.loads(open(rf"{META_DIR}/heart/batch2cat.json", "r").read()),
    "eye": json.loads(open(rf"{META_DIR}/eye/batch2cat.json", "r").read()),
    "gut_fetal": json.loads(open(rf"{META_DIR}/gut_fetal/batch2cat.json", "r").read()),
    "skin": json.loads(open(rf"{META_DIR}/skin/batch2cat.json", "r").read()),
    "COVID": json.loads(open(rf"{META_DIR}/COVID/batch2cat.json", "r").read()),
    "pancreas": json.loads(open(rf"{META_DIR}/pancreas/batch2cat.json", "r").read()),
}

Gender_Related_Genes = [
    "XIST",
    "DDX3Y",
    "EIF1AY",
    "RPS4Y1",
]

ThreePrime_Enriched_Genes = [
    # "HNRNPA1P48",
    "MTRNR2L8",
    "MTRNR2L12",
    "USP18",
    "ARPC1A",
    "LINC00685",
]

FivePrime_Enriched_Genes = [
    "GSTM2",
    "GSTM1",
    "NBPF26",
    "WBP1",
    # "MATR3-1",
    # "HIST1H2BK",
    "RPP21",
    "RPS10-NUDT3",
    "GET4",
    "MRPS24",
    "POMZP3",
    # "AC004922.1",
    "ARF5",
    "TOMM5",
    "TMEFF1",
    "GNG10",
    # "IGF2-1",
    "EEF1G",
    "C11orf98",
    "BSCL2",
    "RBM4",
    "MTG1",
    "SARNP",
    "CDK2AP1",
    "PSMA6",
    "COX16",
    "NGRN",
    "ATP6V0C",
    "PAM16",
    "ALDOA",
    "TUBB3",
    "RNASEK",
    "GABARAP",
    "PLSCR3",
    "NME2",
    "MRPL38",
    "RPL17",
    "UBE2V1",
    "TRAPPC5",
    "HNRNPL",
    "BCKDHA",
    # "SEPT5",
    "MIF",
    "PDXP",
    "GATD3B",
    "U2AF1",
    "MT-ATP8",
    "MT-ND4L",
    "MT-ND6",
]


def inverse_dict(d):
    return {v: k for k, v in d.items()}


if __name__ == "__main__":
    pass
