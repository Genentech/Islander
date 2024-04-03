import Data_Handler as dh
import umap, scanpy as sc, numpy as np, pandas as pd, matplotlib.pyplot as plt


cfg = {"frameon": False, "legend_fontsize": 10, "legend_fontoutline": 2}


def umap_inline_annotations(adata, obsm="X_umap", obs="cell_type", title=""):
    umap_coords = adata.obsm[obsm]

    df = pd.DataFrame(umap_coords, columns=["x", "y"], index=adata.obs.index)
    df["cluster"] = adata.obs[obs]
    centroids = df.groupby("cluster").mean()

    fig, ax = plt.subplots(figsize=(16, 16), dpi=300)
    with plt.rc_context({"figure.figsize": (16, 16), "figure.dpi": (300)}):
        sc.pl.umap(
            adata,
            color=obs,
            title=title,
            show=False,
            legend_loc=None,
            frameon=False,
            ax=ax,
        )

    for cluster, centroid in centroids.iterrows():
        plt.text(centroid["x"], centroid["y"], str(cluster), fontsize=14, ha="center")
    return fig


def _dist_sweep(adata, obsm="GeneCBM", dist=[0.2, 0.3, 0.5], obs="cell_type"):
    print(dist)
    print("\n\n\n")
    for _dist in dist:
        reducer = umap.UMAP(min_dist=_dist)
        embedding = reducer.fit_transform(adata.obsm[obsm])
        adata.obsm["CACHE_%s" % _dist] = embedding

        fig, ax = plt.subplots(figsize=(16, 16))
        sc.pl.embedding(
            adata,
            basis="CACHE_%s" % _dist,
            color=[obs],
            legend_loc="on data",
            frameon=False,
            ncols=1,
            size=77,
            ax=ax,
        )

        # The legend() function of the axes object is used to set the fontsize.
        ax.legend(fontsize=5)

    return fig


def _obsm_sweep(dataset, dist=0.5, hvg=False, obsms=None):
    _suffix = "_hvg" if hvg else ""
    adata = sc.read_h5ad(dh.DATA_EMB_[dataset + _suffix])
    obsms = list(adata.obsm.keys()).copy() if obsms is None else obsms

    for _obsm in obsms:
        print(_obsm)
        _, _dim = adata.obsm[_obsm].shape
        if _dim != 2:
            reducer = umap.UMAP(min_dist=dist)
            embedding = reducer.fit_transform(adata.obsm[_obsm])
            adata.obsm["%s_UMAP" % _obsm] = embedding
            _plot = "%s_UMAP" % _obsm
        else:
            _plot = _obsm
        sc.pl.embedding(
            adata,
            basis=_plot,
            color=[dh.META_[dataset]["celltype"]],
            save="_%s%s.pdf" % (dataset, _suffix),
            **cfg,
        )


def hex_to_rgb(hex_color):
    """Convert a hex color string to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))


def rgb_to_hex(rgb_color):
    """Convert an RGB tuple to a hex color string."""
    return "#" + "".join([f"{int(round(c * 255)):02x}" for c in rgb_color])


def interpolate_hex_colors(start_hex, end_hex, n):
    """Generate `n` evenly spaced colors between `start_hex` and `end_hex`."""

    # Convert start and end colors to RGB
    start_rgb = hex_to_rgb(start_hex)
    end_rgb = hex_to_rgb(end_hex)

    # Generate the intermediate RGB values
    rgb_values = [np.linspace(start, end, n) for start, end in zip(start_rgb, end_rgb)]

    # Convert each RGB set to hex and return the list
    return [rgb_to_hex(rgb) for rgb in zip(*rgb_values)]
