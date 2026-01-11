from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, TypeAlias

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

UnivariateFeatureSelectionFunction: TypeAlias = Callable[..., Any]


class SelectionStrategy(Enum):
    """Strategy for combining features from multiple selection methods."""

    UNION = "union"  # Select features chosen by ANY method
    INTERSECTION = "intersection"  # Select features chosen by ALL methods
    MAJORITY = "majority"  # Select features chosen by >50% of methods
    WEIGHTED = "weighted"  # Select features based on weighted voting


@dataclass
class FeatureScores:
    """
    Container for feature scores from a single selection method.

    Attributes:
    ------------------------------
        method_name: Name of the selection method (e.g., 'mutual_info', 'f_score').
        feature_names: List of feature names in order.
        raw_scores: Original scores from the method.
        normalized_scores: Scores normalized to [0, 1] range.
        selected_features: Features selected based on threshold.
        threshold: Threshold used for selection.
    """

    method_name: str
    feature_names: list[str]
    raw_scores: np.ndarray
    normalized_scores: np.ndarray
    selected_features: list[str] = field(default_factory=list)
    threshold: float = 0.0

    def select_above_threshold(self, threshold: float) -> list[str]:
        """Select features with normalized score >= threshold."""
        self.threshold = threshold
        self.selected_features = [
            name for name, score in zip(self.feature_names, self.normalized_scores) if score >= threshold
        ]
        return self.selected_features

    def select_top_k(self, k: int) -> list[str]:
        """Select top k features by normalized score."""
        indices = np.argsort(self.normalized_scores)[::-1][:k]
        self.selected_features = [self.feature_names[i] for i in indices]
        return self.selected_features

    def select_top_percentile(self, percentile: float) -> list[str]:
        """Select features in the top percentile (e.g., 0.2 = top 20%)."""
        k = max(1, int(len(self.feature_names) * percentile))
        return self.select_top_k(k)

    def asdict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "method_name": self.method_name,
            "feature_names": self.feature_names,
            "raw_scores": self.raw_scores.tolist(),
            "normalized_scores": self.normalized_scores.tolist(),
            "selected_features": self.selected_features,
            "threshold": self.threshold,
        }


@dataclass
class FeatureSelectionResult:
    """
    Result of automated feature selection combining multiple methods.

    Attributes:
    ------------------------------
        method_scores: Dictionary mapping method names to their FeatureScores.
        selected_features: Final list of selected features after combining methods.
        strategy: Strategy used to combine results.
        feature_votes: Dictionary mapping feature names to vote counts.
        feature_rankings: Dictionary mapping feature names to average rank across methods.
    """

    method_scores: dict[str, FeatureScores] = field(default_factory=dict)
    selected_features: list[str] = field(default_factory=list)
    strategy: SelectionStrategy = SelectionStrategy.UNION
    feature_votes: dict[str, int] = field(default_factory=dict)
    feature_rankings: dict[str, float] = field(default_factory=dict)

    def to_df(self) -> pd.DataFrame:
        """
        Create a summary DataFrame showing scores from all methods for each feature.

        Returns:
            DataFrame with features as rows and methods as columns.
        """
        if not self.method_scores:
            return pd.DataFrame()

        # Get all feature names from first method
        first_method = next(iter(self.method_scores.values()))
        feature_names = first_method.feature_names

        data: dict[str, Any] = {"feature": feature_names}
        for method_name, scores in self.method_scores.items():
            data[f"{method_name}_score"] = scores.normalized_scores.tolist()
            data[f"{method_name}_selected"] = [f in scores.selected_features for f in feature_names]

        data["votes"] = [self.feature_votes.get(f, 0) for f in feature_names]
        data["avg_rank"] = [self.feature_rankings.get(f, len(feature_names)) for f in feature_names]
        data["final_selected"] = [f in self.selected_features for f in feature_names]

        result_df = pd.DataFrame(data)
        return result_df.sort_values("avg_rank")


def _select_numerical_features(df: pd.DataFrame, columns: list[str] | None = None) -> list[str]:
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


def _discretize_features(X: pd.DataFrame | pd.Series, n_bins: int) -> np.ndarray:
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

    columns = _select_numerical_features(df, columns)

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

# TODO: move plot-related methods to separate feature_plots.py file and use the results from the computation methods
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

    numerical_columns = _select_numerical_features(df, df.columns.to_list())
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
    X_discrete = _discretize_features(X, n_bins)

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

    numerical_columns = _select_numerical_features(df)
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

    numerical_columns = _select_numerical_features(df)
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

    X_discrete = _discretize_features(X, dconst.DHI_FEATURE_SELECTION_DEFAULT_KBINS_N_BINS)

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

    numerical_columns = _select_numerical_features(df)
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


def compute_correlation_scores(
    df: pd.DataFrame,
    target_column: str,
    label_columns: list[str] | None = None,
    columns: list[str] | None = None,
) -> FeatureScores | None:
    """
    Compute correlation scores between features and the target column.

    Uses absolute Pearson correlation coefficient as the score.

    :param pd.DataFrame df: Input DataFrame.
    :param str target_column: Name of the target column.
    :param list[str] | None label_columns: Columns to exclude from feature selection.
    :param list[str] | None columns: Optional list of feature columns to consider.
    :return FeatureScores | None: FeatureScores object with scores, or None if computation fails.
    """
    if df.empty:
        logger.warning("The dataframe is empty, skipping correlation score computation")
        return None

    # Filter out label columns (except target)
    if label_columns:
        label_columns_to_exclude = list(set(label_columns) - {target_column})
        df = df.drop(columns=[c for c in label_columns_to_exclude if c in df.columns])

    columns = _select_numerical_features(df, columns)
    if not columns:
        logger.warning("No numerical columns found")
        return None

    if target_column not in columns:
        logger.warning(f"Target column '{target_column}' not in numerical columns")
        return None

    feature_columns = [c for c in columns if c != target_column]
    if not feature_columns:
        logger.warning("No feature columns found after excluding target")
        return None

    # Compute correlation with target
    correlations = df[feature_columns].corrwith(df[target_column]).abs()
    raw_scores = np.asarray(correlations.values, dtype=np.float64)
    raw_scores = np.nan_to_num(raw_scores, nan=0.0, posinf=1.0, neginf=0.0)

    # Normalize to [0, 1] (correlations are already in [0, 1] after abs())
    normalized_scores = raw_scores.copy()

    return FeatureScores(
        method_name="correlation",
        feature_names=feature_columns,
        raw_scores=raw_scores,
        normalized_scores=normalized_scores,
    )


def compute_chi2_scores(
    df: pd.DataFrame,
    label_columns: list[str],
    target_column: str,
    n_bins: int = dconst.DHI_FEATURE_SELECTION_DEFAULT_KBINS_N_BINS,
) -> FeatureScores | None:
    """
    Compute chi-squared independence test scores.

    Chi-squared test measures the dependence between categorical variables.
    Features are discretized into bins before the test.

    :param pd.DataFrame df: Input DataFrame.
    :param list[str] label_columns: Columns to exclude from feature selection.
    :param str target_column: Name of the target column.
    :param int n_bins: Number of bins for discretization.
    :return FeatureScores | None: FeatureScores object with chi2 scores, or None if computation fails.
    """
    if df.empty:
        logger.warning("The dataframe is empty, skipping chi2 score computation")
        return None

    label_columns = list(set(label_columns) - {target_column})
    df = df.drop(columns=label_columns)

    numerical_columns = _select_numerical_features(df, df.columns.to_list())
    if not numerical_columns or target_column not in numerical_columns:
        logger.warning("Invalid columns for chi2 computation")
        return None

    X = df[numerical_columns].drop(columns=[target_column])
    if X is None or X.empty:
        return None

    y = df[target_column].astype(int)
    feature_names = X.columns.tolist()

    X_discrete = _discretize_features(X, n_bins)
    chi2_scores, _ = skfs.chi2(X_discrete, y)
    chi2_scores = np.nan_to_num(chi2_scores, nan=0.0, posinf=float(np.finfo(np.float64).max), neginf=0.0)

    # Normalize scores
    max_score = np.max(chi2_scores)
    normalized_scores = chi2_scores / max_score if max_score > 0 else chi2_scores

    return FeatureScores(
        method_name="chi2",
        feature_names=feature_names,
        raw_scores=chi2_scores,
        normalized_scores=normalized_scores,
    )


def compute_variance_scores(
    df: pd.DataFrame,
    label_columns: list[str],
    target_column: str,
) -> FeatureScores | None:
    """
    Compute variance scores for features.

    Higher variance indicates more information content (for non-constant features).

    :param pd.DataFrame df: Input DataFrame.
    :param list[str] label_columns: Columns to exclude from feature selection.
    :param str target_column: Name of the target column.
    :return FeatureScores | None: FeatureScores object with variance scores, or None if computation fails.
    """
    if df.empty:
        logger.warning("The dataframe is empty, skipping variance score computation")
        return None

    label_columns = list(set(label_columns) - {target_column})
    df = df.drop(columns=label_columns)

    numerical_columns = _select_numerical_features(df)
    if not numerical_columns or target_column not in numerical_columns:
        return None

    X = df[numerical_columns].drop(columns=[target_column])
    feature_names = X.columns.tolist()

    variances = np.var(np.asarray(X.values, dtype=np.float64), axis=0)
    variances = np.nan_to_num(variances, nan=0.0, posinf=float(np.finfo(np.float64).max), neginf=0.0)

    # Normalize scores
    max_var = np.max(variances)
    normalized_scores = variances / max_var if max_var > 0 else variances

    return FeatureScores(
        method_name="variance",
        feature_names=feature_names,
        raw_scores=variances,
        normalized_scores=normalized_scores,
    )


def compute_univariate_scores(
    df: pd.DataFrame,
    score_func: UnivariateFeatureSelectionFunction,
    label_columns: list[str],
    target_column: str,
    method_name: str | None = None,
) -> FeatureScores | None:
    """
    Compute univariate feature selection scores using a specified scoring function.

    Supports f_regression, mutual_info_regression, f_classif, mutual_info_classif, etc.

    :param pd.DataFrame df: Input DataFrame.
    :param UnivariateFeatureSelectionFunction score_func: Scoring function from sklearn.feature_selection.
    :param list[str] label_columns: Columns to exclude from feature selection.
    :param str target_column: Name of the target column.
    :param str | None method_name: Optional custom name for the method.
    :return FeatureScores | None: FeatureScores object with computed scores, or None if computation fails.
    """
    if df.empty:
        logger.warning("The dataframe is empty, skipping univariate score computation")
        return None

    label_columns = list(set(label_columns) - {target_column})
    df = df.drop(columns=label_columns)

    numerical_columns = _select_numerical_features(df)
    if not numerical_columns or target_column not in numerical_columns:
        return None

    X = df[numerical_columns].drop(columns=[target_column])
    if X is None or X.empty:
        return None

    y = df[target_column]
    feature_names = X.columns.tolist()

    # For chi2-like functions, discretize the input
    if score_func in (skfs.chi2,):
        X_processed = _discretize_features(X, dconst.DHI_FEATURE_SELECTION_DEFAULT_KBINS_N_BINS)
    else:
        # Convert to float64 to prevent overflow in dot product operations
        X_processed = np.asarray(X.values, dtype=np.float64)

    # Convert y to float64 as well
    y_processed = np.asarray(y.values, dtype=np.float64)

    result = score_func(X_processed, y_processed)
    if isinstance(result, tuple):
        scores = result[0]
    else:
        scores = result

    # Handle nan and inf values - replace inf with large finite values, then cap
    scores = np.nan_to_num(scores, nan=0.0, posinf=float(np.finfo(np.float64).max), neginf=0.0)

    # Normalize scores
    max_score = np.max(scores)
    normalized_scores = scores / max_score if max_score > 0 else scores

    name = method_name if method_name else score_func.__name__

    return FeatureScores(
        method_name=name,
        feature_names=feature_names,
        raw_scores=scores,
        normalized_scores=normalized_scores,
    )


class StatisticalFeatureSelector:
    """
    Automated feature selection combining multiple statistical methods.

    This class provides a unified interface to:
    1. Compute scores from multiple feature selection methods
    2. Select features based on configurable thresholds
    3. Combine results using various strategies (union, intersection, voting)

    Example usage:
        selector = StatisticalFeatureSelector(
            df=df,
            target_column="diabetes_risk_score",
            label_columns=["diabetes_stage", "ethnicity"],
            task_type="regression"
        )
        result = selector.select_features(
            methods=["f_score", "mutual_info", "correlation"],
            threshold=0.2,
            strategy=SelectionStrategy.MAJORITY
        )
        selected_features = result.selected_features
    """

    # Available scoring methods for each task type
    REGRESSION_METHODS = {
        "f_score": skfs.f_regression,
        "mutual_info": skfs.mutual_info_regression,
        "correlation": None,  # Special handling
        "variance": None,  # Special handling
    }

    CLASSIFICATION_METHODS = {
        "f_score": skfs.f_classif,
        "mutual_info": skfs.mutual_info_classif,
        "chi2": skfs.chi2,
        "correlation": None,
        "variance": None,
    }

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        label_columns: list[str],
        task_type: str = "regression",
    ):
        """
        Initialize the StatisticalFeatureSelector.

        :param pd.DataFrame df: Input DataFrame containing features and target.
        :param str target_column: Name of the target column.
        :param list[str] label_columns: List of columns to exclude (non-feature columns).
        :param str task_type: Either "regression" or "classification".
        """
        self.df = df
        self.target_column = target_column
        self.label_columns = label_columns
        self.task_type = task_type

        self._available_methods = self.REGRESSION_METHODS if task_type == "regression" else self.CLASSIFICATION_METHODS
        self._computed_scores: dict[str, FeatureScores] = {}

    @property
    def available_methods(self) -> list[str]:
        """Return list of available method names for the current task type."""
        return list(self._available_methods.keys())

    def compute_feature_scores(self, methods: list[str] | None = None) -> dict[str, FeatureScores]:
        """
        Compute feature scores using specified methods.

        :param list[str] | None methods: List of method names to use. If None, uses all available methods.
        :return dict[str, FeatureScores]: Dictionary mapping method names to FeatureScores objects.
        """
        if methods is None:
            methods = self.available_methods

        for method in methods:
            if method not in self._available_methods:
                logger.warning(f"Unknown method '{method}' for task type '{self.task_type}', skipping")
                continue

            if method in self._computed_scores:
                logger.info(f"Using cached scores for method '{method}'")
                continue

            scores = self._compute_method_scores(method)
            if scores is not None:
                self._computed_scores[method] = scores
                logger.info(f"Computed scores for method '{method}'")

        return self._computed_scores

    def _compute_method_scores(self, method: str) -> FeatureScores | None:
        """Compute scores for a single method."""
        if method == "correlation":
            return compute_correlation_scores(self.df, self.target_column, self.label_columns)
        elif method == "variance":
            return compute_variance_scores(self.df, self.label_columns, self.target_column)
        elif method == "chi2":
            return compute_chi2_scores(self.df, self.label_columns, self.target_column)
        else:
            score_func = self._available_methods.get(method)
            if score_func is not None:
                return compute_univariate_scores(
                    self.df,
                    score_func,
                    self.label_columns,
                    self.target_column,
                    method_name=method,
                )
        return None

    def select_features(
        self,
        methods: list[str] | None = None,
        threshold: float = 0.2,
        strategy: SelectionStrategy = SelectionStrategy.UNION,
        top_k: int | None = None,
        top_percentile: float | None = None,
        method_weights: dict[str, float] | None = None,
        min_votes: int | None = None,
    ) -> FeatureSelectionResult:
        """
        Select features using multiple methods and combine results.

        :param list[str] | None methods: List of method names to use. If None, uses all available.
        :param float threshold: Normalized score threshold for feature selection (0.0 to 1.0).
        :param SelectionStrategy strategy: Strategy for combining results from multiple methods.
        :param int | None top_k: If provided, select top k features instead of using threshold.
        :param float | None top_percentile: If provided, select top percentile of features.
        :param dict[str, float] | None method_weights: Weights for each method (used with WEIGHTED strategy).
        :param int | None min_votes: Minimum votes required (used with custom voting).
        :return FeatureSelectionResult: Containing selected features and analysis.
        """
        # Compute scores for all requested methods
        scores = self.compute_feature_scores(methods)

        if not scores:
            logger.warning("No scores computed, returning empty result")
            return FeatureSelectionResult()

        # Apply selection to each method's scores
        for method_name, feature_scores in scores.items():
            if top_k is not None:
                feature_scores.select_top_k(top_k)
            elif top_percentile is not None:
                feature_scores.select_top_percentile(top_percentile)
            else:
                feature_scores.select_above_threshold(threshold)

        # Combine results using the specified strategy
        result = self._aggregate_selections(scores, strategy, method_weights, min_votes)
        result.strategy = strategy

        return result

    def _aggregate_selections(
        self,
        scores: dict[str, FeatureScores],
        strategy: SelectionStrategy,
        method_weights: dict[str, float] | None = None,
        min_votes: int | None = None,
    ) -> FeatureSelectionResult:
        """Combine feature selections from multiple methods."""
        result = FeatureSelectionResult(method_scores=scores)

        # Get all feature names from ALL methods (union of all feature sets)
        all_features: set[str] = set()
        for method_scores in scores.values():
            all_features.update(method_scores.feature_names)

        # Count votes for each feature
        feature_votes: dict[str, int] = dict.fromkeys(all_features, 0)
        for method_scores in scores.values():
            for feature in method_scores.selected_features:
                if feature in feature_votes:
                    feature_votes[feature] += 1

        result.feature_votes = feature_votes

        # Compute average ranking across methods
        feature_ranks: dict[str, list[int]] = {f: [] for f in all_features}
        for method_scores in scores.values():
            # Rank features by normalized score (higher score = lower rank = better)
            sorted_indices = np.argsort(method_scores.normalized_scores)[::-1]
            for rank, idx in enumerate(sorted_indices):
                feature = method_scores.feature_names[idx]
                if feature in feature_ranks:
                    feature_ranks[feature].append(rank + 1)

        # For features with no rankings, assign worst rank
        max_rank = max(len(m.feature_names) for m in scores.values())
        result.feature_rankings = {
            f: float(np.mean(ranks)) if ranks else float(max_rank) for f, ranks in feature_ranks.items()
        }

        # Select features based on strategy
        n_methods = len(scores)

        if strategy == SelectionStrategy.UNION:
            # Select features chosen by ANY method
            result.selected_features = [f for f, votes in feature_votes.items() if votes > 0]

        elif strategy == SelectionStrategy.INTERSECTION:
            # Select features chosen by ALL methods
            result.selected_features = [f for f, votes in feature_votes.items() if votes == n_methods]

        elif strategy == SelectionStrategy.MAJORITY:
            # Select features chosen by more than half of methods
            min_required = n_methods // 2 + 1
            result.selected_features = [f for f, votes in feature_votes.items() if votes >= min_required]

        elif strategy == SelectionStrategy.WEIGHTED:
            # Weighted voting based on method weights
            if method_weights is None:
                method_weights = dict.fromkeys(scores.keys(), 1.0)

            weighted_scores: dict[str, float] = dict.fromkeys(all_features, 0.0)
            for method_name, method_scores in scores.items():
                weight = method_weights.get(method_name, 1.0)
                for i, feature in enumerate(method_scores.feature_names):
                    weighted_scores[feature] += method_scores.normalized_scores[i] * weight

            # Normalize weighted scores
            max_weighted = max(weighted_scores.values())
            if max_weighted > 0:
                weighted_scores = {f: s / max_weighted for f, s in weighted_scores.items()}

            # Select features above threshold (use average threshold from methods)
            avg_threshold = np.mean([s.threshold for s in scores.values()])
            result.selected_features = [f for f, s in weighted_scores.items() if s >= avg_threshold]

        # Apply minimum votes constraint if specified
        if min_votes is not None:
            result.selected_features = [f for f in result.selected_features if feature_votes[f] >= min_votes]

        # Sort by average ranking
        result.selected_features.sort(key=lambda f: result.feature_rankings.get(f, float("inf")))

        logger.info(f"Selected {len(result.selected_features)} features using {strategy.value} strategy")

        return result

    def plot_scores(
        self,
        methods: list[str] | None = None,
        show_threshold: float | None = None,
    ) -> None:
        """
        Plot feature scores from computed methods.

        :param list[str] | None methods: List of methods to plot. If None, plots all computed.
        :param float | None show_threshold: Optional threshold line to display on plots.
        """
        if not self._computed_scores:
            logger.warning("No scores computed yet. Call compute_scores() first.")
            return

        methods_to_plot = methods if methods else list(self._computed_scores.keys())

        for method in methods_to_plot:
            if method not in self._computed_scores:
                logger.warning(f"Scores for method '{method}' not computed, skipping")
                continue

            scores = self._computed_scores[method]

            fig = px.bar(
                x=scores.feature_names,
                y=scores.normalized_scores,
                title=f"Feature Scores | Method: {method} | Target: {self.target_column}",
                labels={"x": "Features", "y": "Normalized Score"},
                width=dconst.DHI_PLOT_WIDTH,
                height=dconst.DHI_PLOT_HEIGHT,
            )

            # Highlight selected features
            colors = ["green" if f in scores.selected_features else "blue" for f in scores.feature_names]
            fig.update_traces(marker_color=colors, hovertemplate="%{x}: Score %{y:.3f}<extra></extra>")

            if show_threshold is not None:
                fig.add_hline(
                    y=show_threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Threshold: {show_threshold}",
                )

            fig.show()

    def plot_comparison(self, show_threshold: float | None = None) -> None:
        """
        Plot a comparison of scores from all computed methods.

        :param float | None show_threshold: Optional threshold line to display on plots.
        """
        if not self._computed_scores:
            logger.warning("No scores computed yet. Call compute_scores() first.")
            return

        # Get feature names
        first_scores = next(iter(self._computed_scores.values()))
        feature_names = first_scores.feature_names

        fig = go.Figure()

        for method_name, scores in self._computed_scores.items():
            fig.add_trace(
                go.Bar(
                    name=method_name,
                    x=feature_names,
                    y=scores.normalized_scores,
                    hovertemplate=f"{method_name}: %{{y:.3f}}<extra></extra>",
                )
            )

        fig.update_layout(
            title=f"Feature Scores Comparison | Target: {self.target_column}",
            xaxis_title="Features",
            yaxis_title="Normalized Score",
            barmode="group",
            width=dconst.DHI_PLOT_WIDTH,
            height=dconst.DHI_PLOT_HEIGHT,
        )

        if show_threshold is not None:
            fig.add_hline(
                y=show_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {show_threshold}",
            )

        fig.show()

    def get_summary_df(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all computed scores.

        :return pd.DataFrame: DataFrame with features and their scores from each method.
        """
        if not self._computed_scores:
            logger.warning("No scores computed yet. Call compute_scores() first.")
            return pd.DataFrame()

        first_scores = next(iter(self._computed_scores.values()))
        feature_names = first_scores.feature_names

        data: dict[str, Any] = {"feature": feature_names}
        for method_name, scores in self._computed_scores.items():
            data[method_name] = scores.normalized_scores.tolist()

        result_df = pd.DataFrame(data)
        result_df["mean_score"] = result_df.iloc[:, 1:].mean(axis=1)
        return result_df.sort_values("mean_score", ascending=False)


def select_best_features(
    df: pd.DataFrame,
    target_column: str,
    label_columns: list[str],
    task_type: str = "regression",
    methods: list[str] | None = None,
    threshold: float = 0.2,
    strategy: SelectionStrategy = SelectionStrategy.MAJORITY,
    plot: bool = False,
) -> list[str]:
    """
    Convenience function for quick automated statistical feature selection.

    :param pd.DataFrame df: Input DataFrame.
    :param str target_column: Name of the target column.
    :param list[str] label_columns: List of columns to exclude.
    :param str task_type: Either "regression" or "classification".
    :param list[str] | None methods: List of methods to use. Defaults to ["f_score", "mutual_info", "correlation", "variance"].
    :param float threshold: Score threshold for feature selection.
    :param SelectionStrategy strategy: Strategy for combining results.
    :param bool plot: Whether to display plots.
    :return list[str]: List of selected feature names.
    """
    if methods is None:
        methods = ["f_score", "mutual_info", "correlation", "variance"]

    selector = StatisticalFeatureSelector(
        df=df,
        target_column=target_column,
        label_columns=label_columns,
        task_type=task_type,
    )

    result = selector.select_features(
        methods=methods,
        threshold=threshold,
        strategy=strategy,
    )

    if plot:
        selector.plot_scores(show_threshold=threshold)
        selector.plot_comparison(show_threshold=threshold)

    logger.info(f"Auto-selected {len(result.selected_features)} features: {result.selected_features}")

    return result.selected_features
