from typing import Any, Mapping, Optional, TypeAlias

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn.feature_selection as skfs
import skrebate
from sklearn.base import BaseEstimator
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

import dhi.constants as dconst
from dhi.utils import get_logger

logger = get_logger(__name__)

from typing import Callable

UnivariateFeatureSelectionFunction: TypeAlias = Callable[..., Any]


def _select_numerical_columns(df: pd.DataFrame, columns: list[str] | None = None) -> list[str]:
    if not columns:
        logger.warning("No columns provided, using all numerical columns")
        columns = df.select_dtypes(include=["number"]).columns.tolist()
    else:
        logger.info("Filtering columns with DataFrame columns")
        columns = [column for column in columns if column in df.columns]
        columns = df[columns].select_dtypes(include=["number"]).columns.tolist()

    if not columns:
        logger.warning("No numerical columns found, skipping feature selection")
        return []

    return columns


def _discrete_x(X: pd.DataFrame | pd.Series, n_bins: int) -> np.ndarray:
    X_discrete = np.zeros_like(X.values)

    # Discretize every feature independently
    for i in range(X.shape[1]):
        col_values = np.asarray(X.iloc[:, i].values)
        if np.min(col_values) < 0:
            col_values = col_values - np.min(col_values)

        kb = KBinsDiscretizer(
            n_bins=n_bins,
            encode="ordinal",
            strategy="uniform",
        )
        X_discrete[:, i] = kb.fit_transform(col_values.reshape(-1, 1)).ravel()

    return X_discrete


def correlation_matrix(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    title: str = "Correlation Heatmap",
) -> None:
    """
    Plots the correlation matrix of the dataframe.

    :param pd.DataFrame df: The dataframe to plot the correlation matrix of
    :param list[str] | None columns: The columns to plot the correlation matrix of, defaults to None
    :param str title: The title of the correlation matrix plot, defaults to "Correlation Heatmap"
    """
    if df.empty:
        logger.warning("The dataframe is empty, skipping correlation matrix plot")
        return

    if not columns:
        logger.warning("No columns provided, using all numerical columns")
        columns = df.select_dtypes(include=["number"]).columns.tolist()
    else:
        logger.info("Filtering columns with DataFrame columns")
        columns = [column for column in columns if column in df.columns]
        columns = df[columns].select_dtypes(include=["number"]).columns.tolist()

    if not columns:
        logger.warning("No numerical columns found, skipping correlation matrix plot")
        return

    logger.info(f"Plotting correlation matrix for {columns} out of {len(df.columns)} columns")

    df_subset = df[columns]
    if df_subset is None or df_subset.empty:
        logger.warning("No numerical columns found, skipping correlation matrix plot")
        return
    if isinstance(df_subset, pd.Series):
        df_subset = df_subset.to_frame()
    df_corr = df_subset.corr()

    logger.info(f"Using title: {title}")
    fig = px.imshow(
        df_corr,
        labels={"x": "Features", "y": "Features"},
        text_auto=True,
        aspect="auto",
        title=title,
        color_continuous_scale=dconst.DHI_PLOT_COLOR_CONTINUOUS_SCALE,
        width=dconst.DHI_PLOT_WIDTH,
        height=dconst.DHI_PLOT_HEIGHT,
        zmin=-1,
        zmax=1,
    )

    fig.update_traces(hovertemplate="%{x}: %{y}: Correlation%{z:.2f}<extra></extra>")
    fig.show()

    logger.info("Plotting of correlation matrix completed successfully")


def chi2_independence_test(
    df: pd.DataFrame,
    label_columns: list[str],
    target_column: str,
    n_bins: int = dconst.DHI_FEATURE_SELECTION_DEFAULT_KBINS_N_BINS,
) -> None:
    """
    Performs a chi2 independence test on the dataframe.

    Removes label columns from the dataframe, except for the target column.

    :param pd.DataFrame df: The dataframe to perform the chi2 independence test on
    :param list[str] label_columns: The columns to remove from the dataframe
    :param str target_column: The column to perform the chi2 independence test on
    :param int n_bins: The number of bins to use for the discretization, defaults to DHI_FEATURE_SELECTION_DEFAULT_KBINS_N_BINS
    """
    if df.empty:
        logger.warning("The dataframe is empty, skipping chi2 independence test")
        return

    label_columns = list(set(label_columns) - {target_column})
    df = df.drop(columns=label_columns)

    numerical_columns = _select_numerical_columns(df, df.columns.to_list())
    if not numerical_columns:
        logger.warning("No numerical columns found, skipping chi2 independence test")
        return
    if target_column not in numerical_columns:
        logger.warning("Target column is not a numerical column, skipping chi2 independence test")
        return

    if n_bins <= 0:
        logger.warning("Number of bins must be greater than 0, using default number of bins")
        n_bins = dconst.DHI_FEATURE_SELECTION_DEFAULT_KBINS_N_BINS

    X = df[numerical_columns].drop(columns=[target_column])
    if X is None or X.empty:
        logger.error("X is None or empty, skipping chi2 independence test")
        return

    y = df[target_column].astype(int)

    feature_names = X.columns.tolist()

    # Discretize every feature independently
    X_discrete = _discrete_x(X, n_bins)

    logger.info(f"Performing chi2 independence test on {df.columns} out of {len(df.columns)} columns")
    chi2_scores, p_values = skfs.chi2(X_discrete, y)

    logger.info(f"Chi2 scores: {chi2_scores}")
    logger.info(f"P values: {p_values}")

    fig = px.bar(
        x=feature_names,
        y=chi2_scores,
        title=f"Chi2 Independence Test | Target: {target_column}",
        labels={"x": "Features", "y": "Chi2 Scores"},
        width=dconst.DHI_PLOT_WIDTH,
        height=dconst.DHI_PLOT_HEIGHT,
    )
    fig.update_traces(hovertemplate="%{x}: Chi2 Scores%{y:.2f}<extra></extra>")
    fig.show()

    logger.info("Plotting of chi2 independence test completed successfully")


def variance_threshold_feature_selection(
    df: pd.DataFrame,
    label_columns: list[str],
    target_column: str,
    threshold: float = dconst.DHI_FEATURE_SELECTION_DEFAULT_VARIANCE_THRESHOLD,
) -> np.ndarray:
    """
    Performs a variance threshold feature selection on the dataframe.

    Selects only those features that have a variance greater than the threshold.

    :param pd.DataFrame df: The dataframe to perform the variance threshold feature selection on
    :param list[str] label_columns: The columns to remove from the dataframe
    :param str target_column: The column to perform the variance threshold feature selection on
    :param float threshold: The threshold to perform the variance threshold feature selection on, defaults to dconst.DHI_FEATURE_SELECTION_DEFAULT_VARIANCE_THRESHOLD
    :return np.ndarray: The selected features
    """
    if df.empty:
        logger.warning("The dataframe is empty, skipping variance threshold feature selection")
        return np.array([])

    label_columns = list(set(label_columns) - {target_column})
    df = df.drop(columns=label_columns)

    numerical_columns = _select_numerical_columns(df)
    if not numerical_columns:
        logger.warning("No numerical columns found, skipping variance threshold feature selection")
        return np.array([])
    if target_column not in numerical_columns:
        logger.warning("Target column is not a numerical column, skipping variance threshold feature selection")
        return np.array([])

    X = df[numerical_columns].drop(columns=[target_column])
    y = df[target_column]

    selector = skfs.VarianceThreshold(threshold=threshold)
    selector.fit(X, y)

    high_var_indices = selector.get_support(indices=True)

    # Calculate variances before and after filtering
    variances_original = np.var(np.asarray(X.values), axis=0)
    x_filtered = X.iloc[:, high_var_indices]

    variances_filtered = np.var(x_filtered.values, axis=0)
    logger.info(f"Variances of filtered features: {variances_filtered}")

    # Get feature names for original and filtered features
    original_features = list(X.columns)
    selected_features = (
        [original_features[i] for i in high_var_indices] if high_var_indices is not None else original_features
    )

    # Create bar plot visualization
    fig = go.Figure()

    # Add original variances bar
    fig.add_trace(
        go.Bar(
            x=original_features,
            y=variances_original,
            name="Original",
            marker_color="blue",
            hovertemplate="Feature: %{x}<br>Variance: %{y:.4f}<extra></extra>",
        )
    )

    # Add filtered variances bar (only for selected features)
    fig.add_trace(
        go.Bar(
            x=selected_features,
            y=variances_filtered,
            name="Filtered",
            marker_color="red",
            hovertemplate="Feature: %{x}<br>Variance: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Variance of Features before and after Filtering | Threshold: {threshold}",
        xaxis_title="Feature",
        yaxis_title="Variance",
        width=dconst.DHI_PLOT_WIDTH,
        height=dconst.DHI_PLOT_HEIGHT,
        legend={"title": "Dataset"},
        hovermode="closest",
    )

    fig.show()
    logger.info("Plotting of variance threshold feature selection completed successfully")

    return np.array(x_filtered.columns)


def univariate_feature_selection(
    df: pd.DataFrame,
    score_func: UnivariateFeatureSelectionFunction,
    label_columns: list[str],
    target_column: str,
    mode: str = "percentile",
    params: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Performs a univariate feature selection on the dataframe.

    Allows different modes and settable parameters for the feature selection.

    :param pd.DataFrame df: The dataframe to perform the univariate feature selection on
    :param UnivariateFeatureSelectionFunction score_func: The score function to perform the univariate feature selection on
    :param list[str] label_columns: The columns to remove from the dataframe
    :param str target_column: The column to perform the univariate feature selection on
    :param str mode: The mode to perform the univariate feature selection on, defaults to "percentile"
    :param Optional[Mapping[str, Any]] params: The parameters to perform the univariate feature selection on, defaults to None
    """
    if df.empty:
        logger.warning("The dataframe is empty, skipping univariate feature selection")
        return

    label_columns = list(set(label_columns) - {target_column})
    df = df.drop(columns=label_columns)

    numerical_columns = _select_numerical_columns(df)
    if not numerical_columns:
        logger.warning("No numerical columns found, skipping univariate feature selection")
        return
    if target_column not in numerical_columns:
        logger.warning("Target column is not a numerical column, skipping univariate feature selection")
        return

    X = df[numerical_columns].drop(columns=[target_column])
    if X is None or X.empty:
        logger.error("X is None or empty, skipping univariate feature selection")
        return

    y = df[target_column]

    feature_names = X.columns.tolist()

    X_discrete = _discrete_x(X, dconst.DHI_FEATURE_SELECTION_DEFAULT_KBINS_N_BINS)

    if mode not in dconst.DHI_FEATURE_SELECTION_MODES:
        logger.warning(f"Invalid mode: {mode}, using default mode: {dconst.DHI_FEATURE_SELECTION_DEFAULT_MODE}")
        mode = dconst.DHI_FEATURE_SELECTION_DEFAULT_MODE

    params = dconst.DHI_FEATURE_SELECTION_MODES[mode] if params is None else params

    selector = skfs.GenericUnivariateSelect(score_func=score_func, mode=mode)  # pyright: ignore[reportArgumentType]
    selector.set_params(**params)

    selector.fit(X_discrete, y)
    scores = np.nan_to_num(selector.scores_, nan=0.0)
    scores /= np.max(scores)

    logger.info(f"Univariate feature selection scores: {scores}")

    fig = px.bar(
        x=feature_names,
        y=scores,
        title=f"Univariate Feature Selection | Target: {target_column} | Score Function: {score_func.__name__}",
        labels={"x": "Features", "y": "Scores"},
        width=dconst.DHI_PLOT_WIDTH,
        height=dconst.DHI_PLOT_HEIGHT,
    )
    fig.update_traces(hovertemplate="%{x}: Scores%{y:.2f}<extra></extra>")
    fig.show()

    logger.info("Plotting of univariate feature selection completed successfully")


def model_feature_selection(
    clf: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    prefit: bool = False,
) -> tuple[pd.DataFrame, BaseEstimator]:
    """
    Performs a model feature selection on the dataframe.

    :param BaseEstimator clf: The classifier to perform the model feature selection on
    :param pd.DataFrame X: The dataframe to perform the model feature selection on
    :param pd.Series y: The series to perform the model feature selection on
    :param bool prefit: Whether the classifier is already fitted, defaults to False
    :return tuple[pd.DataFrame, BaseEstimator]: The input dataframe with the selected features and the selector object
    """
    if not prefit:
        logger.warning("Classifier is not fitted, fitting it")
        clf.fit(X, y)  # pyright: ignore[reportAttributeAccessIssue]
        prefit = True

    selector = skfs.SelectFromModel(clf, prefit=prefit)
    x_new = np.asarray(selector.transform(X))

    logger.info(f"Selected {x_new.shape} features out of {len(X.columns)} features")

    return (
        pd.DataFrame(np.asarray(x_new), columns=selector.get_feature_names_out()),
        selector,
    )


def relief_feature_selection(
    df: pd.DataFrame,
    label_columns: list[str],
    target_column: str,
    n_features: int = dconst.DHI_FEATURE_SELECTION_DEFAULT_RELIEF_N_FEATURES,
) -> None:
    """
    Performs a relief feature selection on the dataframe.

    :param pd.DataFrame df: The dataframe to perform the relief feature selection on
    :param list[str] label_columns: The columns to remove from the dataframe
    :param str target_column: The column to perform the relief feature selection on
    :param int n_features: The number of features to select, defaults to DHI_FEATURE_SELECTION_DEFAULT_RELIEF_N_FEATURES
    :return skrebate.ReliefF: The selector object
    """
    if df.empty:
        logger.warning("The dataframe is empty, skipping relief feature selection")
        return

    label_columns = list(set(label_columns) - {target_column})
    df = df.drop(columns=label_columns)

    numerical_columns = _select_numerical_columns(df)
    if not numerical_columns:
        logger.warning("No numerical columns found, skipping relief feature selection")
        return
    if target_column not in numerical_columns:
        logger.warning("Target column is not a numerical column, skipping relief feature selection")
        return

    # Reset index to ensure sequential indexing (required by ReliefF)
    df = df.reset_index(drop=True)

    X = df[numerical_columns].drop(columns=[target_column])
    if X is None or X.empty:
        logger.error("X is None or empty, skipping relief feature selection")
        return

    y = df[target_column]

    feature_names = X.columns.tolist()

    X_array = X.values.astype(np.float64)
    y_array = np.asarray(y.values)

    if not np.issubdtype(y_array.dtype, np.integer):
        logger.info("Converting target variable to integer labels for ReliefF")
        le = LabelEncoder()
        y_array = np.asarray(le.fit_transform(y_array))

    logger.info(f"Performing ReliefF feature selection X={X_array.shape} and y={y_array.shape}")
    selector = skrebate.ReliefF(n_features_to_select=n_features, n_jobs=-1)
    selector = selector.fit(X_array, y_array)

    logger.info(f"Selected {n_features} features out of {len(feature_names)} features")
    logger.info(f"Selected features: {selector.top_features_}")
