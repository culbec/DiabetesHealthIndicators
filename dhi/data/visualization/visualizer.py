import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Iterable

import dhi.constants as dconst
from dhi.utils import get_logger

logger = get_logger(__name__)


def plot2d(x: np.ndarray, y: np.ndarray, labels: Iterable[str] | None = None, title: str = "2D Plot", hover_data: dict = {}) -> None:
    """
    Plots the 2D plot of the dataframe.

    :param np.ndarray x: The array to plot the 2D plot of
    :param np.ndarray y: The array to plot the 2D plot of
    :param Iterable[str] labels: The labels to use for the plot, defaults to None
    :param str title: The title of the 2D plot, defaults to "2D Plot"
    :param dict hover_data: The data to display on hover, defaults to {}
    """
    if x.shape[0] != y.shape[0]:
        logger.warning("The arrays have different lengths, skipping 2D plot")
        return

    if labels is not None:
        color = labels
    else:
        color = None

    logger.info(f"Plotting 2D plot for {x.shape} with title: {title}")
    fig = px.scatter(
        x=x,
        y=y,
        title=title,
        color=color,
        color_continuous_scale=dconst.DHI_PLOT_COLOR_CONTINUOUS_SCALE,
        width=dconst.DHI_PLOT_WIDTH,
        height=dconst.DHI_PLOT_HEIGHT,
        hover_data=hover_data,
    )
    fig.show()
    logger.info("Plotting of 2D plot completed successfully")


def plot3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, labels: Iterable[str] | None = None, title: str = "3D Plot", hover_data: dict = {}) -> None:
    """
    Plots the 3D plot of the dataframe.

    :param np.ndarray x: The array to plot the 3D plot of
    :param np.ndarray y: The array to plot the 3D plot of
    :param np.ndarray z: The array to plot the 3D plot of
    :param Iterable[str] labels: The labels to use for the plot, defaults to None
    :param str title: The title of the 3D plot, defaults to "3D Plot"
    :param dict hover_data: The data to display on hover, defaults to {}
    """
    if not (x.shape[0] == y.shape[0] == z.shape[0]):
        logger.warning("The arrays have different lengths, skipping 3D plot")
        return

    if labels is not None:
        color = labels
    else:
        color = None

    logger.info(f"Plotting 3D plot for {x.shape} with title: {title}")
    fig = px.scatter_3d(
        x=x,
        y=y,
        z=z,
        color=color,
        title=title,
        color_continuous_scale=dconst.DHI_PLOT_COLOR_CONTINUOUS_SCALE,
        width=dconst.DHI_PLOT_WIDTH,
        height=dconst.DHI_PLOT_HEIGHT,
        hover_data=hover_data,
    )
    fig.show()
    logger.info("Plotting of 3D plot completed successfully")


def plot_boxplots(df: pd.DataFrame, columns_to_plot: list[str] = [], title: str = "Boxplots") -> None:
    """
    Plots boxplots for the specified columns of the provided DataFrame.

    :param pd.DataFrame df: The dataframe to plot the boxplots of
    :param list[str] columns_to_plot: The columns to plot. If empty, will plot all numerical columns if fallback_to_numerical is True.
    :param str title: The title of the plot. Defaults to 'Boxplots'
    """
    if df.empty:
        logger.warning("DataFrame is empty. Cannot plot boxplots.")
        return

    if not columns_to_plot:
        logger.warning("No columns specified for boxplots. Selecting all numerical columns.")
        df_columns_to_plot = df.select_dtypes(include=["number"]).columns.tolist()
    else:
        df_columns_to_plot = [col for col in columns_to_plot if col in df.columns]
        logger.info(f"Filtered columns to plot: {df_columns_to_plot}. Selecting only the numerical ones.")
        df_columns_to_plot = df[df_columns_to_plot].select_dtypes(include=["number"]).columns.tolist()

    if not df_columns_to_plot:
        logger.warning("No numerical columns available for boxplots. Skipping.")
        return

    logger.info(f"Plotting boxplots for columns: {df_columns_to_plot}")

    plot_data = df[df_columns_to_plot].astype("float32")

    fig = px.box(data_frame=plot_data, title=title)
    fig.update_layout(xaxis_title="Features", yaxis_title="Values", boxmode="group")
    fig.show()


def plot_histograms(
    df: pd.DataFrame,
    columns_to_plot: list[str] = [],
    bins: int = dconst.DHI_PLOT_HISTOGRAM_DEFAULT_BINS,
    title: str = "Histogram Plot",
    cols_per_subplot: int = dconst.DHI_PLOT_SUBPLOT_COLS_PER_ROW,
) -> None:
    """
    Plots histograms for the specified columns of the provided DataFrame in a single image.

    :param pd.DataFrame df: The dataframe to plot the histograms of
    :param list[str] columns_to_plot: The columns to plot, defaults to []
    :param int bins: The number of bins to squeeze the unique values into, defaults to DHI_PLOT_HISTOGRAM_DEFAULT_BINS.
        If the bins are larger than the number of unique values, or lower than 1, will assign the bins to the value of unique values.
    :param str title: The title of the plot, defaults to 'Histogram Plot'.
    :param int cols_per_subplot: Number of columns per row in the subplot grid, defaults to DHI_PLOT_SUBPLOT_COLS_PER_ROW.
    """
    if df.empty:
        logger.warning("DataFrame is empty. Cannot plot histograms.")
        return

    if not columns_to_plot:
        logger.warning("No columns specified for histograms. Selecting all numerical columns.")
        df_columns_to_plot = df.select_dtypes(include=["number"]).columns.tolist()
    else:
        df_columns_to_plot = [col for col in columns_to_plot if col in df.columns]
        logger.info(
            f"Filtered columns to plot for histograms: {df_columns_to_plot}. Selecting only the numerical ones."
        )
        df_columns_to_plot = df[df_columns_to_plot].select_dtypes(include=["number"]).columns.tolist()

    if not df_columns_to_plot:
        logger.warning("No numerical columns available for histograms. Skipping.")
        return

    logger.info(f"Plotting histograms for columns: {df_columns_to_plot}")

    bins = bins if bins > 0 else dconst.DHI_PLOT_HISTOGRAM_DEFAULT_BINS
    n_cols = cols_per_subplot
    n_rows = -(-len(df_columns_to_plot) // n_cols)  # Ceiling division

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=df_columns_to_plot,
    )
    histogram_marker = dconst.DHI_PLOT_HISTOGRAM_MARKER

    for idx, col in enumerate(df_columns_to_plot):
        row = (idx // n_cols) + 1
        col_pos = (idx % n_cols) + 1

        unique_values = df[col].nunique()
        effective_bins = min(bins, unique_values)  # Adjust bins if there are fewer unique values than bins

        fig.add_trace(
            go.Histogram(
                x=df[col],
                nbinsx=effective_bins,
                name=col,
                showlegend=False,
                marker=histogram_marker,
            ),
            row=row,
            col=col_pos,
        )

    fig.update_layout(
        title_text=title,
        height=dconst.DHI_PLOT_HISTOGRAM_HEIGHT * n_rows,
        width=dconst.DHI_PLOT_HISTOGRAM_WIDTH * n_cols,
        showlegend=False,
    )
    fig.show()


def plot_distplots(
    df: pd.DataFrame,
    columns_to_plot: list[str] = [],
    title: str = "Feature Distributions",
    cols_per_subplot: int = dconst.DHI_PLOT_SUBPLOT_COLS_PER_ROW,
) -> None:
    """
    Plots distribution plots for the specified columns of the provided DataFrame in a single image.

    :param pd.DataFrame df: The dataframe to plot the distplots of
    :param list[str] columns_to_plot: The columns to plot, defaults to []
    :param str title: The title of the plot, defaults to "Feature Distributions".
    :param int cols_per_subplot: The number of columns per subplot, defaults to DHI_PLOT_SUBPLOT_COLS_PER_ROW.
    """
    if df.empty:
        logger.warning("DataFrame is empty. Cannot plot distribution plots.")
        return

    if not columns_to_plot:
        logger.warning("No columns specified for boxplots. Selecting all numerical columns.")
        df_columns_to_plot = df.select_dtypes(include=["number"]).columns.tolist()
    else:
        df_columns_to_plot = [col for col in columns_to_plot if col in df.columns]
        logger.info(f"Filtered columns to plot: {df_columns_to_plot}. Selecting only the numerical ones.")
        df_columns_to_plot = df[df_columns_to_plot].select_dtypes(include=["number"]).columns.tolist()

    if not df_columns_to_plot:
        logger.warning("No numerical columns available for distribution plots. Skipping.")
        return

    logger.info(f"Plotting distribution plots for columns: {df_columns_to_plot}")

    n_cols = cols_per_subplot
    n_rows = -(-len(df_columns_to_plot) // n_cols)  # Ceiling division

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=df_columns_to_plot,
    )
    for idx, col in enumerate(df_columns_to_plot):
        row = (idx // n_cols) + 1
        col_pos = (idx % n_cols) + 1

        hist_data = [df[col].dropna()]
        group_labels = [col]
        distplot_fig = ff.create_distplot(
            hist_data,
            group_labels=group_labels,
        )

        for trace in distplot_fig["data"]:
            fig.add_trace(trace, row=row, col=col_pos)
    fig.update_layout(
        title_text=title,
        height=dconst.DHI_PLOT_HISTOGRAM_HEIGHT * n_rows,
        width=dconst.DHI_PLOT_HISTOGRAM_WIDTH * n_cols,
        showlegend=False,
    )
    fig.show()


def plot_by_index(df: pd.DataFrame, columns_to_plot: list[str] = [], title: str = "Index-based Plot") -> None:
    """
    Plots the specified columns of the provided DataFrame against its index.

    :param pd.DataFrame df: The dataframe to plot the by index of
    :param list[str] columns_to_plot: The columns to plot, defaults to []
    :param str title: The title of the plot, defaults to 'Index-based Plot'.
    """
    if df.empty:
        logger.warning("DataFrame is empty. Cannot plot index-based.")
        return

    if not columns_to_plot:
        logger.warning("No columns specified for index-based plot. Selecting all numerical columns.")
        df_columns_to_plot = df.select_dtypes(include=["number"]).columns.tolist()
    else:
        df_columns_to_plot = [col for col in columns_to_plot if col in df.columns]
        logger.info(f"Filtered columns to plot: {df_columns_to_plot}. Selecting only the numerical ones.")
        df_columns_to_plot = df[df_columns_to_plot].select_dtypes(include=["number"]).columns.tolist()

    if not df_columns_to_plot:
        logger.warning("No numerical columns available for index-based plot. Skipping.")
        return

    logger.info(f"Plotting index-based for columns: {df_columns_to_plot}")
    fig = px.line(df, x=df.index, y=columns_to_plot, title=title)
    fig.update_layout(xaxis_title="Index", yaxis_title="Values", legend_title="Features")
    fig.show()
