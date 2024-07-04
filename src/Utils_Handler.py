import torch, numpy as np, scanpy as sc, pandas as pd

from tqdm import tqdm
from datetime import datetime
from torch.autograd import grad
from scipy.stats import trim_mean
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

_isnan = lambda x: x != x
mean_ = lambda x: sum(x) / len(x)
DATETIME = datetime.now().strftime("%Y_%m_%d_%H_%M")
filtered_ = lambda a_tuple: tuple(x for x in a_tuple if x is not None)


def norm2count(rec_):
    normed_counts = np.exp(rec_) - 1
    size_factors = 1e4 / normed_counts.sum(axis=1)
    return np.log(normed_counts * size_factors[:, None] + 1)


def preprocess(adata, top_n_cells=100, min_cells=5, min_genes=500, min_counts=1000):
    # Sanity Check
    subset = adata.X[:, :top_n_cells].toarray()
    non_negative = np.all(subset >= 0)
    integer_values = np.all(subset.astype(int) == subset)
    assert non_negative and integer_values

    # Quality Control
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print("=" * 77)
    print(rf"{adata.n_vars} genes x {adata.n_obs} cells after quality control.")
    print("=" * 77)

    # Pre-Processing
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    return adata


def umaps_rawcounts(adata, n_neighbors=10, n_pcs=50):
    _list = list(adata.obsm.keys())
    for _obsm in _list:
        if "Provided_" in _obsm:
            continue
        adata.obsm["Provided_" + _obsm.lower()] = adata.obsm[_obsm]
        del adata.obsm[_obsm]

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata)
    return adata


# import scipy.sparse
def compare_sparse_matrices(a, b):
    """
    Compares two sparse matrices for equality.
    Parameters:
    a (csr_matrix or csc_matrix): The first sparse matrix.
    b (csr_matrix or csc_matrix): The second sparse matrix.
    Returns:
    bool: True if the matrices are equal, False otherwise.
    """
    # Ensure the matrices have the same format for comparison
    a_csr = a.tocsr()
    b_csr = b.tocsr()
    # Compare data, indices, and indptr attributes
    data_equal = np.array_equal(a_csr.data, b_csr.data)
    indices_equal = np.array_equal(a_csr.indices, b_csr.indices)
    indptr_equal = np.array_equal(a_csr.indptr, b_csr.indptr)
    # Optionally, compare shapes
    shape_equal = a_csr.shape == b_csr.shape
    return data_equal and indices_equal and indptr_equal and shape_equal


def check_unique(adata, batch, metas):
    for meta in metas:
        print(meta, "=" * 77)
        for batch_id in tqdm(adata.obs[batch].unique()):
            adata_batch = adata[adata.obs[batch] == batch_id]
            if adata_batch.obs[meta].unique().__len__() > 1:
                print(batch_id, adata_batch.obs[meta].unique())
                break
        print("\n")


def benchmark_obsm(adata, batch_key, label_key, obsm_keys):
    biocons = BioConservation(nmi_ari_cluster_labels_leiden=True)
    batcorr = BatchCorrection()

    benchmarker = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=obsm_keys,
        bio_conservation_metrics=biocons,
        batch_correction_metrics=batcorr,
        pre_integrated_embedding_obsm_key="X_pca",
        n_jobs=-1,
    )
    benchmarker.prepare(neighbor_computer=None)
    benchmarker.benchmark()

    return benchmarker.get_results(min_max_scale=False)


def calculate_centroids(X, labels):
    centroids = dict()
    for label in labels.unique():
        centroids[label] = np.mean(X[labels == label], axis=0)
    return centroids


def calculate_trimmed_means(X, labels, trim_proportion=0.2, ignore_=[]):
    centroids = dict()
    if isinstance(X, csr_matrix):
        X = X.toarray()
    for label in labels.unique():
        if label in ignore_:
            continue
        centroids[label] = trim_mean(X[labels == label], proportiontocut=trim_proportion, axis=0)
    return centroids


def compute_classwise_distances(centroids):
    centroid_vectors = np.array([centroids[key] for key in sorted(centroids.keys())])
    distances = cdist(centroid_vectors, centroid_vectors, "euclidean")
    return pd.DataFrame(distances, columns=sorted(centroids.keys()), index=sorted(centroids.keys()))
