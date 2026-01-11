import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

import dhi.constants as dconst
from dhi.utils import get_logger
from dhi.data.visualization.exploratory_plots import plot2d, plot3d

logger = get_logger(__name__)

# TODO: find a better way to separate the statistical computation logic and the plotting logic in a dimensionality_plots.py file
def _component_reduction(df: pd.DataFrame, color_column: str, n_components: int, title: str, method: str) -> None:
    """
    Helper method to perform component reduction on the provided dataframe.

    :param pd.DataFrame df: The dataframe to perform component reduction on
    :param str color_column: The column to color the plot by
    :param int n_components: The number of components to perform component reduction on
    :param str title: The title of the component reduction plot
    :param str method: The method to perform component reduction on
    """
    data_for_reduction = df.select_dtypes(include=["number", "category"])

    match method:
        case "pca":
            pca = PCA(n_components=n_components)

            logger.info(f"Fitting component reduction model with {data_for_reduction.columns} columns")

            reduced_data = pca.fit_transform(data_for_reduction)

            logger.info(f"Explained variance ratios by PCA components: {pca.explained_variance_ratio_.round(4)}")
            logger.info(
                f"Total explained variance by first {n_components} PCA components: {pca.explained_variance_ratio_.sum().round(4)}"
            )
        case "tsne":
            tsne = TSNE(n_components=n_components)

            logger.info(f"Fitting component reduction model with {data_for_reduction.columns} columns")

            reduced_data = tsne.fit_transform(data_for_reduction)

        case "umap":
            umap = UMAP(n_components=n_components)

            logger.info(f"Fitting component reduction model with {data_for_reduction.columns} columns")

            reduced_data = umap.fit_transform(data_for_reduction)
        case _:
            logger.warning(f"Unsupported method: {method}, skipping component reduction")
            return

    if reduced_data is None or not hasattr(reduced_data, "shape"):
        logger.warning("Component reduction failed, skipping plot")
        return

    reduced_data = np.asarray(reduced_data)

    logger.info(f"Reduced data has {reduced_data.shape[1]} components using '{method}' method")

    hover_data = {
        (data_for_reduction.index.name or "Index"): data_for_reduction.index,
        **{col: data_for_reduction[col] for col in data_for_reduction.columns},
    }

    match n_components:
        case 2:
            x = reduced_data[:, 0]
            y = reduced_data[:, 1]
            plot2d(
                x,
                y,
                labels=df[color_column].values,
                title=title + f" (target={color_column})",
                hover_data=hover_data,
            )
        case 3:
            x = reduced_data[:, 0]
            y = reduced_data[:, 1]
            z = reduced_data[:, 2]
            plot3d(
                x,
                y,
                z,
                labels=df[color_column].values,
                title=title + f" (target={color_column})",
                hover_data=hover_data,
            )
        case _:
            logger.warning(f"Unsupported number of components: {n_components}, skipping PCA plot")
            return

    logger.info("Plotting of component reduction completed successfully")


def plot_pca(
    df: pd.DataFrame,
    color_column: str,
    n_components: int = dconst.DHI_COMPONENT_REDUCTION_DEFAULT_N_COMPONENTS,
    title: str = "PCA Plot",
) -> None:
    """
    PCA component reduction plot on the provided dataframe.

    :param pd.DataFrame df: The dataframe to plot the PCA plot of
    :param str color_column: The column to color the plot by
    :param int n_components: The number of components to plot, defaults to DHI_COMPONENT_REDUCTION_DEFAULT_N_COMPONENTS
    :param str title: The title of the PCA plot, defaults to "PCA Plot"
    """
    if df.empty:
        logger.warning("The dataframe is empty, skipping PCA plot")
        return

    _component_reduction(df, color_column, n_components, title, "pca")


def plot_tsne(
    df: pd.DataFrame,
    color_column: str,
    n_components: int = dconst.DHI_COMPONENT_REDUCTION_DEFAULT_N_COMPONENTS,
    title: str = "TSNE Plot",
) -> None:
    """
    TSNE component reduction plot on the provided dataframe.

    :param pd.DataFrame df: The dataframe to plot the TSNE plot of
    :param str color_column: The column to color the plot by
    :param int n_components: The number of components to plot, defaults to DHI_COMPONENT_REDUCTION_DEFAULT_N_COMPONENTS
    :param str title: The title of the TSNE plot, defaults to "TSNE Plot"
    """
    if df.empty:
        logger.warning("The dataframe is empty, skipping TSNE plot")
        return

    _component_reduction(df, color_column, n_components, title, "tsne")


def plot_umap(
    df: pd.DataFrame,
    color_column: str,
    n_components: int = dconst.DHI_COMPONENT_REDUCTION_DEFAULT_N_COMPONENTS,
    title: str = "UMAP Plot",
) -> None:
    """
    UMAP component reduction plot on the provided dataframe.

    :param pd.DataFrame df: The dataframe to plot the UMAP plot of
    :param str color_column: The column to color the plot by
    :param int n_components: The number of components to plot, defaults to DHI_COMPONENT_REDUCTION_DEFAULT_N_COMPONENTS
    :param str title: The title of the UMAP plot, defaults to "UMAP Plot"
    """
    if df.empty:
        logger.warning("The dataframe is empty, skipping UMAP plot")
        return

    _component_reduction(df, color_column, n_components, title, "umap")
