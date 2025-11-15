from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn.feature_selection as skfs
import skrebate
from sklearn.base import BaseEstimator
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

import dhi.const as dconst
import dhi.statistics.const as dstatconst
from dhi.utils import get_logger

logger = get_logger(__name__)


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
    df_corr = df[columns].corr()

    logger.info(f"Using title: {title}")
    fig = px.imshow(
        df_corr,
        labels=dict(x="Features", y="Features"),
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
    n_bins: int = dstatconst.DHI_FEATURE_SELECTION_DEFAULT_KBINS_N_BINS,
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

    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if not numerical_columns:
        logger.warning("No numerical columns found, skipping chi2 independence test")
        return
    if not target_column in numerical_columns:
        logger.warning("Target column is not a numerical column, skipping chi2 independence test")
        return

    if n_bins <= 0:
        logger.warning("Number of bins must be greater than 0, using default number of bins")
        n_bins = dstatconst.DHI_FEATURE_SELECTION_DEFAULT_KBINS_N_BINS

    x = df[numerical_columns].drop(columns=[target_column])
    feature_names = x.columns.tolist()

    if (x < 0).any().any():
        logger.warning("Some features have negative values, discretizing them")
        kbins = KBinsDiscretizer(n_bins=dstatconst.DHI_FEATURE_SELECTION_DEFAULT_KBINS_N_BINS)
        x = kbins.fit_transform(x)
    else:
        x = x.values

    y = df[target_column]

    logger.info(f"Performing chi2 independence test on {df.columns} out of {len(df.columns)} columns")
    chi2_scores, p_values = skfs.chi2(x, y)

    logger.info(f"Chi2 scores: {chi2_scores}")
    logger.info(f"P values: {p_values}")

    fig = px.bar(
        x=feature_names,
        y=chi2_scores,
        title=f"Chi2 Independence Test | Target: {target_column}",
        labels=dict(x="Features", y="Chi2 Scores"),
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
    threshold: float = dstatconst.DHI_FEATURE_SELECTION_DEFAULT_VARIANCE_THRESHOLD,
) -> None:
    """
    Performs a variance threshold feature selection on the dataframe.

    :param pd.DataFrame df: The dataframe to perform the variance threshold feature selection on
    :param list[str] label_columns: The columns to remove from the dataframe
    :param str target_column: The column to perform the variance threshold feature selection on
    :param float threshold: The threshold to perform the variance threshold feature selection on, defaults to dstatconst.DHI_FEATURE_SELECTION_DEFAULT_VARIANCE_THRESHOLD
    """
    if df.empty:
        logger.warning("The dataframe is empty, skipping variance threshold feature selection")
        return

    label_columns = list(set(label_columns) - {target_column})
    df = df.drop(columns=label_columns)

    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if not numerical_columns:
        logger.warning("No numerical columns found, skipping variance threshold feature selection")
        return
    if not target_column in numerical_columns:
        logger.warning("Target column is not a numerical column, skipping variance threshold feature selection")
        return

    x = df[numerical_columns].drop(columns=[target_column])

    selector = skfs.VarianceThreshold(threshold=threshold)
    x_new = selector.fit_transform(x)

    # Calculate variances before and after filtering
    variances_original = np.var(x.values, axis=0)
    variances_filtered = np.var(x_new, axis=0)

    # Get feature names for original and filtered features
    original_features = list(x.columns)
    selected_features = selector.get_feature_names_out(x.columns)

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
        legend=dict(title="Dataset"),
        hovermode="closest",
    )

    fig.show()
    logger.info("Plotting of variance threshold feature selection completed successfully")


def univariate_feature_selection(
    df: pd.DataFrame,
    label_columns: list[str],
    target_column: str,
    mode: str = "percentile",
    params: dict[str, Any] = {},
) -> None:
    """
    Performs a univariate feature selection on the dataframe.

    Allows different modes and settable parameters for the feature selection.

    :param pd.DataFrame df: The dataframe to perform the univariate feature selection on
    :param list[str] label_columns: The columns to remove from the dataframe
    :param str target_column: The column to perform the univariate feature selection on
    :param str mode: The mode to perform the univariate feature selection on, defaults to "percentile"
    :param dict[str, Any] params: The parameters to perform the univariate feature selection on, defaults to {}
    :param list[str] | None columns: The columns to perform the univariate feature selection on, defaults to None
    """
    if df.empty:
        logger.warning("The dataframe is empty, skipping univariate feature selection")
        return

    label_columns = list(set(label_columns) - {target_column})
    df = df.drop(columns=label_columns)

    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if not numerical_columns:
        logger.warning("No numerical columns found, skipping univariate feature selection")
        return
    if not target_column in numerical_columns:
        logger.warning("Target column is not a numerical column, skipping univariate feature selection")
        return

    x = df[numerical_columns].drop(columns=[target_column])
    y = df[target_column]

    if mode not in dstatconst.DHI_FEATURE_SELECTION_MODES:
        logger.warning(f"Invalid mode: {mode}, using default mode: {dstatconst.DHI_FEATURE_SELECTION_DEFAULT_MODE}")
        mode = dstatconst.DHI_FEATURE_SELECTION_DEFAULT_MODE

    params = dstatconst.DHI_FEATURE_SELECTION_MODES[mode] if not params else params

    selector = skfs.GenericUnivariateSelect(score_func=skfs.f_classif, mode=mode) # pyright: ignore[reportArgumentType]
    selector.set_params(**params)

    selector.fit(x, y)

    # Handle zero p-values to avoid log10(0) = -inf
    # Replace zero p-values with a very small value (machine epsilon)
    pvalues = np.asarray(selector.pvalues_).copy()
    pvalues[pvalues == 0] = np.finfo(float).eps

    scores = -np.log10(pvalues)

    # Normalize scores, handling edge cases
    max_score = scores.max()
    if max_score > 0 and np.isfinite(max_score):
        scores /= max_score
    elif max_score == 0:
        # All p-values are 1, so all scores are 0 - no normalization needed
        logger.warning("All p-values are 1, scores are all zero")
    else:
        # Handle any remaining non-finite values (shouldn't happen after epsilon replacement, but just in case)
        finite_scores = scores[np.isfinite(scores)]
        if len(finite_scores) > 0:
            max_finite = finite_scores.max()
            if max_finite > 0:
                scores = np.where(np.isfinite(scores), scores / max_finite, 1.0)
            else:
                scores = np.where(np.isfinite(scores), scores, 0.0)
        else:
            # All scores are non-finite (shouldn't happen), set to 1.0
            scores = np.ones_like(scores)
            logger.warning("All scores are non-finite, setting to 1.0")

    logger.info(f"Univariate feature selection scores: {scores}")

    fig = px.bar(
        x=x.columns,
        y=scores,
        title=f"Univariate Feature Selection | Target: {target_column}",
        labels=dict(x="Features", y="Scores"),
        width=dconst.DHI_PLOT_WIDTH,
        height=dconst.DHI_PLOT_HEIGHT,
    )
    fig.update_traces(hovertemplate="%{x}: Scores%{y:.2f}<extra></extra>")
    fig.show()

    logger.info("Plotting of univariate feature selection completed successfully")

def model_feature_selection(
    clf: BaseEstimator,
    x: pd.DataFrame,
    y: pd.Series,
    prefit: bool = False,
) -> tuple[pd.DataFrame, BaseEstimator]:
    """
    Performs a model feature selection on the dataframe.

    :param BaseEstimator clf: The classifier to perform the model feature selection on
    :param pd.DataFrame x: The dataframe to perform the model feature selection on
    :param pd.Series y: The series to perform the model feature selection on
    :param bool prefit: Whether the classifier is already fitted, defaults to False
    :return tuple[pd.DataFrame, BaseEstimator]: The input dataframe with the selected features and the selector object
    """
    if not prefit:
        logger.warning("Classifier is not fitted, fitting it")
        clf.fit(x, y) # pyright: ignore[reportAttributeAccessIssue]
        prefit = True

    selector = skfs.SelectFromModel(clf, prefit=prefit)
    x_new = selector.transform(x)

    logger.info(f"Selected {x_new.shape} features out of {len(x.columns)} features")

    return pd.DataFrame(np.asarray(x_new), columns=selector.get_feature_names_out()), selector


def relief_feature_selection(
    df: pd.DataFrame,
    label_columns: list[str],
    target_column: str,
    n_features: int = dstatconst.DHI_FEATURE_SELECTION_DEFAULT_RELIEF_N_FEATURES,
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

    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if not numerical_columns:
        logger.warning("No numerical columns found, skipping relief feature selection")
        return
    if not target_column in numerical_columns:
        logger.warning("Target column is not a numerical column, skipping relief feature selection")
        return

    # Reset index to ensure sequential indexing (required by ReliefF)
    df = df.reset_index(drop=True)

    x = df[numerical_columns].drop(columns=[target_column])
    feature_names = x.columns.tolist()
    y = df[target_column]

    x_array = x.values.astype(np.float64)
    y_array = np.asarray(y.values)

    if not np.issubdtype(y_array.dtype, np.integer):
        logger.info("Converting target variable to integer labels for ReliefF")
        le = LabelEncoder()
        y_array = np.asarray(le.fit_transform(y_array))

    logger.info(f"Performing ReliefF feature selection x={x_array.shape} and y={y_array.shape}")
    selector = skrebate.ReliefF(n_features_to_select=n_features, n_jobs=-1)
    selector = selector.fit(x_array, y_array)

    logger.info(f"Selected {n_features} features out of {len(feature_names)} features")
    logger.info(f"Selected features: {selector.top_features_}")
