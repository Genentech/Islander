from __future__ import absolute_import, division, print_function
import sys, time, pandas as pd, numpy as np, scanpy as sc, Data_Handler as dh, Utils_Handler as uh
from tqdm import tqdm


def rank_diff_old(df1, df2):
    """=== Deprecated ==="""
    # Pairwise drop NaNs
    paired_non_nan = pd.concat([df1, df2], axis=1).dropna()
    df1_ranked, df2_ranked = (
        paired_non_nan.iloc[:, 0].rank(),
        paired_non_nan.iloc[:, 1].rank(),
    )
    # df1_ranked, df2_ranked = df1.rank(), df2.rank()
    rank_difference = abs(df1_ranked - df2_ranked)
    return rank_difference.mean().mean()


def rank_diff(df1, df2):
    spearman_corr = {}
    for col in df1.columns:
        paired_non_nan = pd.concat([df1[col], df2[col]], axis=1).dropna()
        spearman_corr[col] = paired_non_nan.iloc[:, 0].corr(paired_non_nan.iloc[:, 1], method="spearman")

    return pd.DataFrame.from_dict(spearman_corr, orient="index", columns=["Spearman Correlation"])


def corr_diff(df1, df2):
    pearson_corr = {}
    # spearman_corr = {}
    for col in df1.columns:
        # Pairwise drop NaNs
        paired_non_nan = pd.concat([df1[col], df2[col]], axis=1).dropna()
        pearson_corr[col] = paired_non_nan.iloc[:, 0].corr(paired_non_nan.iloc[:, 1], method="pearson")
        # spearman_corr[col] = paired_non_nan.iloc[:, 0].corr(paired_non_nan.iloc[:, 1], method="spearman")

    return pd.DataFrame.from_dict(pearson_corr, orient="index", columns=["Pearson Correlation"])


def corrw_diff(df1, df2):
    pearson_corr = {}
    for col in df1.columns:
        # Pairwise drop NaNs
        paired_non_nan = pd.concat([df1[col], df2[col]], axis=1).dropna()
        pearson_corr[col] = weighted_pearson(paired_non_nan.iloc[:, 0], paired_non_nan.iloc[:, 1], paired_non_nan.iloc[:, 1])

    return pd.DataFrame.from_dict(pearson_corr, orient="index", columns=["Pearson Correlation"])


def weighted_pearson(x, y, distances):
    # Prevent division by zero and control the influence of very small distances
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = 1 / distances
        weights[distances == 0] = 0  # Assign a zero weight for zero distances

    # Normalize the weights to sum to 1
    weights /= np.sum(weights)

    # Calculate the weighted means
    weighted_mean_x = np.average(x, weights=weights)
    weighted_mean_y = np.average(y, weights=weights)

    # Calculate the weighted covariance and the weighted variances
    covariance = np.sum(weights * (x - weighted_mean_x) * (y - weighted_mean_y))
    variance_x = np.sum(weights * (x - weighted_mean_x) ** 2)
    variance_y = np.sum(weights * (y - weighted_mean_y) ** 2)

    # Calculate the weighted Pearson correlation coefficient
    weighted_pearson_corr = covariance / np.sqrt(variance_x * variance_y)
    return weighted_pearson_corr


if __name__ == "__main__":
    THRESHOLD_BATCH = 100
    THRESHOLD_CELLTYPE = 10
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        dataset = "lung_fetal_donor"
        # dataset = "skin"
    batch_key = dh.META_[dataset]["batch"]
    label_key = dh.META_[dataset]["celltype"]
    adata = sc.read(dh.DATA_EMB_[dataset], first_column_names=True)
    bdata = sc.read(dh.DATA_EMB_[dataset + "_hvg"], first_column_names=True)
    for _obsm in bdata.obsm:
        adata.obsm[_obsm + "_hvg"] = bdata.obsm[_obsm]

    _collect_count_, _collect_pca_ = dict(), dict()
    start_time = time.time()

    ignore_ = []
    for celltype in adata.obs[label_key].unique():
        if adata.obs[label_key].value_counts()[celltype] < THRESHOLD_CELLTYPE:
            print("Ignored %s" % celltype)
            ignore_.append(celltype)
        if "Mesenchymal" in celltype:
            print("Ignored %s" % celltype)
            ignore_.append(celltype)

    for BATCH_ in tqdm(adata.obs[batch_key].unique()):
        adata_batch = adata[adata.obs[batch_key] == BATCH_].copy()

        # === recompute PCA of each batch ===
        if len(adata_batch) < THRESHOLD_BATCH:
            continue

        # NOTE: it didn't make a difference
        # if adata_batch.layers["raw_counts"] is not None:
        #     adata_batch.X = adata_batch.layers["raw_counts"].copy()
        #     sc.pp.normalize_per_cell(adata_batch, counts_per_cell_after=1e4)
        #     del adata_batch.uns["log1p"]
        #     sc.pp.log1p(adata_batch)
        #     # if "log1p" in adata.uns_keys():
        #     #     logg.warning("adata.X seems to be already log-transformed.")
        sc.pp.highly_variable_genes(adata_batch, n_top_genes=1000)
        sc.pp.pca(adata_batch, n_comps=10, use_highly_variable=True)

        centroids_pca = uh.calculate_trimmed_means(
            adata_batch.obsm["X_pca"],
            adata_batch.obs[label_key],
            trim_proportion=0.2,
            ignore_=ignore_,
        )
        pca_pairdist = uh.compute_classwise_distances(centroids_pca)
        _norm_pca = pca_pairdist.div(pca_pairdist.max(axis=0), axis=1)
        del centroids_pca, pca_pairdist

        centroids_count = uh.calculate_trimmed_means(
            adata_batch.X,
            adata_batch.obs[label_key],
            trim_proportion=0.2,
            ignore_=ignore_,
        )  # normalised, log1p transformed
        count_pairdist = uh.compute_classwise_distances(centroids_count)
        _norm_count = count_pairdist.div(count_pairdist.max(axis=0), axis=1)
        del centroids_count, count_pairdist

        _collect_count_[BATCH_], _collect_pca_[BATCH_] = _norm_count, _norm_pca

    df_combined = pd.concat(_collect_count_.values(), axis=0, sort=False)
    concensus_df_count = df_combined.groupby(df_combined.index).mean()
    concensus_df_count = concensus_df_count.loc[concensus_df_count.columns, :]
    concensus_df_count = concensus_df_count / concensus_df_count.max(axis=0)

    df_combined = pd.concat(_collect_pca_.values(), axis=0, sort=False)
    concensus_df_pca = df_combined.groupby(df_combined.index).mean()
    concensus_df_pca = concensus_df_pca.loc[concensus_df_pca.columns, :]
    concensus_df_pca = concensus_df_pca / concensus_df_pca.max(axis=0)

    # TODO: to add a faliure warning if half of the concensus_df_pca are empty

    # NOTE: there are indeed some NaNs in the concensus_df_count, concensus_df_pca,
    # because there might not exist a batch including both (cell type A, cell type Y), so

    def adata_concensus(adata, obsm, label_key):
        _centroid = uh.calculate_trimmed_means(
            np.array(adata.obsm[obsm]),
            adata.obs[label_key],
            trim_proportion=0.2,
            ignore_=ignore_,
        )
        _pairdist = uh.compute_classwise_distances(_centroid)
        return _pairdist.div(_pairdist.max(axis=0), axis=1)

    res_df = pd.DataFrame(columns=["Rank-Count", "Corr-Count", "Rank-PCA", "Corr-PCA"])
    _obsm_list = list(adata.obsm)
    _obsm_list.sort()

    # adata_df, concensus_df_count, concensus_df_pca are all normalised Euclidean distances
    for _obsm in _obsm_list:
        adata_df = adata_concensus(adata, _obsm, label_key)
        _row_df = pd.DataFrame(
            {
                "Rank-Count": rank_diff(adata_df, concensus_df_count).mean().values,
                "Corr-Count": corr_diff(adata_df, concensus_df_count).mean().values,
                #
                "Rank-PCA": rank_diff(adata_df, concensus_df_pca).mean().values,
                "Corr-PCA": corr_diff(adata_df, concensus_df_pca).mean().values,
                #
                "Corr-Weights": corrw_diff(adata_df, concensus_df_pca).mean().values,
            },
            index=[_obsm],
        )
        res_df = pd.concat([res_df, _row_df], axis=0, sort=False)
    # res_df.to_csv(rf"{dh.RES_DIR}/scGraph/{dataset}_spearman.csv")
    # res_df.to_csv(rf"{dh.RES_DIR}/scGraph/{dataset}_subset1.csv")
    res_df.to_csv(rf"{dh.RES_DIR}/scGraph/{dataset}_new.csv")
    print(res_df)
