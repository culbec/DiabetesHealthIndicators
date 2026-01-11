"""
Statistical metrics for cross-validation analysis.

This module provides statistical analysis tools for evaluating machine learning
model performance across cross-validation folds. It includes descriptive statistics,
confidence intervals, normality testing, and hypothesis testing for model comparison.

The module is designed to work with k-fold cross-validation results, where the
number of samples (folds) is typically small (5-10), requiring appropriate
statistical methods that account for small sample sizes.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


@dataclass
class DescriptiveStats:
    """
    Descriptive statistics computed from cross-validation fold scores.

    Provides a comprehensive summary of the distribution of scores across folds,
    including measures of central tendency, dispersion, and quartile information.

    Attributes:
    ------------------------------
        mean: Arithmetic mean of fold scores.
        std: Sample standard deviation (using Bessel's correction, ddof=1).
        std_error: Standard error of the mean (std / sqrt(n)).
        median: Middle value when scores are sorted.
        min: Minimum fold score.
        max: Maximum fold score.
        iqr: Interquartile range (Q3 - Q1), measures spread of middle 50%.
        q1: First quartile (25th percentile).
        q3: Third quartile (75th percentile).
        coefficient_of_variation: Relative variability as percentage (std/mean * 100).
        n_samples: Number of folds used in computation.
    """

    mean: float
    std: float
    std_error: float
    median: float
    min: float
    max: float
    iqr: float
    q1: float
    q3: float
    coefficient_of_variation: float
    n_samples: int


@dataclass
class ConfidenceInterval:
    """
    Confidence interval for a population parameter estimate.

    Represents the range within which the true population mean is expected
    to fall with a specified probability (confidence level).

    Attributes:
    ------------------------------
        lower: Lower bound of the confidence interval.
        upper: Upper bound of the confidence interval.
        confidence_level: Probability level (e.g., 0.95 for 95% CI).
        point_estimate: The sample mean used as the center of the interval.
        margin_of_error: Half-width of the interval (upper - point_estimate).
    """

    lower: float
    upper: float
    confidence_level: float
    point_estimate: float
    margin_of_error: float


@dataclass
class NormalityTestResult:
    """
    Result of a statistical test for normality of the data distribution.

    Contains the test statistic, p-value, and a boolean indicating whether
    the null hypothesis (data is normally distributed) should be rejected.

    Attributes:
    ------------------------------
        statistic: The Shapiro-Wilk test statistic W (closer to 1 = more normal).
        p_value: Probability of observing this statistic if data is normal.
        is_normal: True if p_value > alpha (fail to reject normality).
        alpha: Significance level used for the test decision.
    """

    statistic: float
    p_value: float
    is_normal: bool
    alpha: float


@dataclass
class HypothesisTestResult:
    """
    Result of a hypothesis test comparing two samples or testing a hypothesis.

    Includes the test statistic, p-value, significance decision, and optionally
    an effect size measure to quantify practical significance.

    Attributes:
    ------------------------------
        test_name: Identifier for the test performed (e.g., "paired_t_test").
        statistic: The computed test statistic value.
        p_value: Probability of observing this result under the null hypothesis.
        is_significant: True if p_value < alpha (reject null hypothesis).
        alpha: Significance level used for the test decision.
        effect_size: Optional effect size measure (e.g., Cohen's d).
        effect_size_interpretation: Qualitative interpretation of effect size.
    """

    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float
    effect_size: Optional[float] = None
    effect_size_interpretation: Optional[str] = None

    def asdict(self) -> Dict[str, Any]:
        """
        Convert to dictionary, excluding None optional fields.

        Required because effect_size fields should be omitted when not computed.
        """
        result = asdict(self)
        # Remove None optional fields for cleaner serialization
        if self.effect_size is None:
            del result["effect_size"]
            del result["effect_size_interpretation"]
        return result


@dataclass
class CVFoldStatistics:
    """
    Comprehensive statistical analysis of cross-validation fold scores.

    Aggregates descriptive statistics, confidence intervals, and normality
    test results into a single container for convenient access and reporting.

    Attributes:
    ------------------------------
        fold_scores: Original list of scores from each CV fold.
        descriptive: Descriptive statistics computed from the fold scores.
        confidence_interval: Confidence interval for the mean score.
        normality_test: Shapiro-Wilk normality test result (None if n < 3).
    """

    fold_scores: List[float]
    descriptive: DescriptiveStats
    confidence_interval: ConfidenceInterval
    normality_test: Optional[NormalityTestResult] = None

    def asdict(self) -> Dict[str, Any]:
        """
        Convert to dictionary, excluding normality_test when None.

        Required because normality_test is not computed when n < 3 folds.
        """
        result = asdict(self)
        # Remove None optional field for cleaner serialization
        if self.normality_test is None:
            del result["normality_test"]
        return result


@dataclass
class ModelComparisonResult:
    """
    Result of statistical comparison between two machine learning models.

    Contains summary statistics for both models, the difference in their
    means, and results from both parametric and non-parametric hypothesis tests.

    Attributes:
    ------------------------------
        model_a_name: Display name for the first model.
        model_b_name: Display name for the second model.
        model_a_mean: Mean CV score for the first model.
        model_b_mean: Mean CV score for the second model.
        mean_difference: Difference in means (model_a_mean - model_b_mean).
        paired_ttest: Results from the paired t-test (parametric).
        wilcoxon_test: Results from Wilcoxon signed-rank test (non-parametric).
    """

    model_a_name: str
    model_b_name: str
    model_a_mean: float
    model_b_mean: float
    mean_difference: float
    paired_ttest: HypothesisTestResult
    wilcoxon_test: Optional[HypothesisTestResult] = None

    def asdict(self) -> Dict[str, Any]:
        """
        Convert to dictionary, excluding wilcoxon_test when None.

        Required because Wilcoxon test is not computed when n < 6 folds.
        Also handles nested HypothesisTestResult that may have None fields.
        """
        result: Dict[str, Any] = {
            "model_a_name": self.model_a_name,
            "model_b_name": self.model_b_name,
            "model_a_mean": self.model_a_mean,
            "model_b_mean": self.model_b_mean,
            "mean_difference": self.mean_difference,
            "paired_ttest": self.paired_ttest.asdict(),
        }
        if self.wilcoxon_test is not None:
            result["wilcoxon_test"] = self.wilcoxon_test.asdict()
        return result


def compute_descriptive_stats(scores: ArrayLike) -> DescriptiveStats:
    """
    Compute descriptive statistics for a set of cross-validation fold scores.

    Calculates measures of central tendency (mean, median), dispersion
    (standard deviation, IQR, range), and relative variability (CV).

    :param ArrayLike scores: Array-like of numeric fold scores.
    :return DescriptiveStats: Dataclass containing all computed statistics.
    :raises ValueError: If the scores array is empty.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    n = len(scores_arr)

    if n == 0:
        raise ValueError("Cannot compute statistics for empty scores array")

    # Central tendency
    mean = float(np.mean(scores_arr))
    median = float(np.median(scores_arr))

    # Dispersion measures
    # Use Bessel's correction (ddof=1) for unbiased sample standard deviation
    std = float(np.std(scores_arr, ddof=1)) if n > 1 else 0.0
    std_error = std / np.sqrt(n) if n > 0 else 0.0

    # Quartiles and interquartile range
    q1 = float(np.percentile(scores_arr, 25))
    q3 = float(np.percentile(scores_arr, 75))
    iqr = q3 - q1

    # Coefficient of variation: relative standard deviation as percentage
    # Guard against division by zero when mean is near zero
    cv = (std / abs(mean) * 100) if abs(mean) > 1e-10 else 0.0

    return DescriptiveStats(
        mean=mean,
        std=std,
        std_error=std_error,
        median=median,
        min=float(np.min(scores_arr)),
        max=float(np.max(scores_arr)),
        iqr=iqr,
        q1=q1,
        q3=q3,
        coefficient_of_variation=cv,
        n_samples=n,
    )


def compute_confidence_interval(
    scores: ArrayLike,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """
    Compute a confidence interval for the mean using Student's t-distribution.

    Uses the t-distribution rather than the normal distribution because CV
    typically has small sample sizes (5-10 folds). The t-distribution has
    heavier tails, providing wider and more accurate intervals for small n.

    The formula is: CI = mean +/- t_(alpha/2, n-1) * (std / sqrt(n))

    :param ArrayLike scores: Array-like of numeric fold scores.
    :param float confidence_level: Desired confidence level, default 0.95 (95% CI).
    :return ConfidenceInterval: Dataclass with bounds and margin of error.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    n = len(scores_arr)

    # Handle edge case: cannot compute CI with fewer than 2 samples
    if n < 2:
        mean = float(np.mean(scores_arr)) if n > 0 else 0.0
        return ConfidenceInterval(
            lower=mean,
            upper=mean,
            confidence_level=confidence_level,
            point_estimate=mean,
            margin_of_error=0.0,
        )

    mean = float(np.mean(scores_arr))
    std = float(np.std(scores_arr, ddof=1))
    std_error = std / np.sqrt(n)

    # Compute t-critical value for two-tailed test
    # For 95% CI, we need the 97.5th percentile (1 - 0.05/2)
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)

    margin_of_error = t_critical * std_error

    return ConfidenceInterval(
        lower=mean - margin_of_error,
        upper=mean + margin_of_error,
        confidence_level=confidence_level,
        point_estimate=mean,
        margin_of_error=margin_of_error,
    )


def test_normality(scores: ArrayLike, alpha: float = 0.05) -> Optional[NormalityTestResult]:
    """
    Test whether fold scores follow a normal distribution using Shapiro-Wilk.

    The Shapiro-Wilk test is recommended for small sample sizes (n < 50) and
    is particularly appropriate for CV results. The null hypothesis is that
    the data is normally distributed.

    :param ArrayLike scores: Array-like of numeric fold scores.
    :param float alpha: Significance level for the test, default 0.05.
    :return Optional[NormalityTestResult]: NormalityTestResult if n >= 3, None otherwise.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    n = len(scores_arr)

    # Shapiro-Wilk test requires at least 3 samples
    if n < 3:
        return None

    # Handle degenerate case: all values identical (zero variance)
    # Constant data trivially satisfies normality
    if np.all(scores_arr == scores_arr[0]):
        return NormalityTestResult(
            statistic=1.0,
            p_value=1.0,
            is_normal=True,
            alpha=alpha,
        )

    # Perform Shapiro-Wilk test
    shapiro_res = stats.shapiro(scores_arr)

    # Extract results from the named tuple returned by scipy
    statistic: float = float(cast(float, shapiro_res[0]))
    p_value: float = float(cast(float, shapiro_res[1]))

    return NormalityTestResult(
        statistic=statistic,
        p_value=p_value,
        is_normal=p_value > alpha,
        alpha=alpha,
    )


def analyze_cv_fold_scores(
    fold_scores: ArrayLike,
    confidence_level: float = 0.95,
    normality_alpha: float = 0.05,
) -> CVFoldStatistics:
    """
    Perform comprehensive statistical analysis of cross-validation fold scores.

    This is the main entry point for analyzing CV results. It combines
    descriptive statistics, confidence interval estimation, and normality
    testing into a single convenient function call.

    :param ArrayLike fold_scores: Array-like of scores from each CV fold.
    :param float confidence_level: Confidence level for interval estimation.
    :param float normality_alpha: Significance level for normality test.
    :return CVFoldStatistics: Containing all analysis results.
    """
    scores_list = list(np.asarray(fold_scores, dtype=np.float64))

    descriptive = compute_descriptive_stats(scores_list)
    ci = compute_confidence_interval(scores_list, confidence_level)
    normality = test_normality(scores_list, normality_alpha)

    return CVFoldStatistics(
        fold_scores=scores_list,
        descriptive=descriptive,
        confidence_interval=ci,
        normality_test=normality,
    )


def _compute_cohens_dz(scores_a: np.ndarray, scores_b: np.ndarray) -> Tuple[float, str]:
    """
    Compute Cohen's d effect size for paired samples.

    Cohen's d measures the standardized difference between two means.
    For paired data, it uses the standard deviation of the differences.
    This quantifies practical significance beyond statistical significance.

    Interpretation thresholds:
        |d| < 0.2:  negligible effect
        0.2 <= |d| < 0.5: small effect
        0.5 <= |d| < 0.8: medium effect
        |d| >= 0.8: large effect

    :param np.ndarray scores_a: Scores from the first model.
    :param np.ndarray scores_b: Scores from the second model.
    :return Tuple[float, str]: Tuple of (effect_size, interpretation_string).
    """
    diff = scores_a - scores_b
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    # Handle edge case: zero variance in differences
    if std_diff < 1e-10:
        d = 0.0 if abs(mean_diff) < 1e-10 else np.inf * np.sign(mean_diff)
    else:
        d = mean_diff / std_diff

    # Interpret effect size magnitude
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return float(d), interpretation


def compare_models(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    alpha: float = 0.05,
) -> ModelComparisonResult:
    """
    Statistically compare two models using their cross-validation fold scores.

    Performs both parametric (paired t-test) and non-parametric (Wilcoxon
    signed-rank) hypothesis tests. The paired t-test assumes normality of
    differences; when this assumption is violated, rely on the Wilcoxon test.

    Both models must have been evaluated on the same CV folds to enable
    paired comparison, which controls for fold-to-fold variability.

    :param ArrayLike scores_a: Fold scores from the first model.
    :param ArrayLike scores_b: Fold scores from the second model.
    :param str model_a_name: Display name for the first model.
    :param str model_b_name: Display name for the second model.
    :param float alpha: Significance level for hypothesis tests.
    :return ModelComparisonResult: With test statistics and conclusions.
    :raises ValueError: If score arrays have different lengths or n < 2.
    """
    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)

    # Validate input: paired comparison requires equal lengths
    if len(arr_a) != len(arr_b):
        raise ValueError(
            f"Score arrays must have same length for paired comparison. Got {len(arr_a)} and {len(arr_b)}"
        )

    n = len(arr_a)
    if n < 2:
        raise ValueError("Need at least 2 fold scores for comparison")

    # Compute summary statistics
    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))
    mean_diff = mean_a - mean_b

    # Paired t-test: tests if mean difference is significantly different from zero
    # Assumes differences are normally distributed
    ttest_res = stats.ttest_rel(arr_a, arr_b)
    t_stat: float = float(cast(float, ttest_res[0]))
    t_pvalue: float = float(cast(float, ttest_res[1]))

    # Compute effect size to assess practical significance
    cohens_dz, effect_interpretation = _compute_cohens_dz(arr_a, arr_b)

    paired_ttest = HypothesisTestResult(
        test_name="paired_t_test",
        statistic=t_stat,
        p_value=t_pvalue,
        is_significant=t_pvalue < alpha,
        alpha=alpha,
        effect_size=cohens_dz,
        effect_size_interpretation=effect_interpretation,
    )

    # Wilcoxon signed-rank test: non-parametric alternative
    # Does not assume normality; based on ranks of absolute differences
    # Requires at least 6 samples for meaningful results
    wilcoxon_result = None
    if n >= 6:
        try:
            diff = arr_a - arr_b
            # Wilcoxon requires non-zero differences to compute ranks
            if not np.all(diff == 0):
                wilcoxon_res = stats.wilcoxon(arr_a, arr_b)
                w_stat: float = float(cast(float, wilcoxon_res[0]))
                w_pvalue: float = float(cast(float, wilcoxon_res[1]))
                wilcoxon_result = HypothesisTestResult(
                    test_name="wilcoxon_signed_rank",
                    statistic=w_stat,
                    p_value=w_pvalue,
                    is_significant=w_pvalue < alpha,
                    alpha=alpha,
                )
        except (ValueError, IndexError, TypeError):
            # Wilcoxon can fail with degenerate data
            pass

    return ModelComparisonResult(
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        model_a_mean=mean_a,
        model_b_mean=mean_b,
        mean_difference=mean_diff,
        paired_ttest=paired_ttest,
        wilcoxon_test=wilcoxon_result,
    )


def format_cv_statistics_report(cv_stats: CVFoldStatistics, metric_name: str = "score") -> str:
    """
    Format cross-validation statistics as a human-readable text report.

    Generates a structured report containing all descriptive statistics,
    confidence intervals, and normality test results in an easy-to-read format.

    :param CVFoldStatistics cv_stats: CVFoldStatistics object to format.
    :param str metric_name: Name of the metric for the report header.
    :return str: Multi-line formatted string report.
    """
    lines = [
        f"Cross-Validation Statistics for '{metric_name}'",
        "=" * 50,
        "",
        "Descriptive Statistics:",
        f"  Mean:      {cv_stats.descriptive.mean:.6f}",
        f"  Std Dev:   {cv_stats.descriptive.std:.6f}",
        f"  Std Error: {cv_stats.descriptive.std_error:.6f}",
        f"  Median:    {cv_stats.descriptive.median:.6f}",
        f"  Min:       {cv_stats.descriptive.min:.6f}",
        f"  Max:       {cv_stats.descriptive.max:.6f}",
        f"  IQR:       {cv_stats.descriptive.iqr:.6f} "
        f"(Q1={cv_stats.descriptive.q1:.6f}, Q3={cv_stats.descriptive.q3:.6f})",
        f"  CV:        {cv_stats.descriptive.coefficient_of_variation:.2f}%",
        f"  N folds:   {cv_stats.descriptive.n_samples}",
        "",
        f"Confidence Interval ({cv_stats.confidence_interval.confidence_level*100:.0f}%):",
        f"  [{cv_stats.confidence_interval.lower:.6f}, {cv_stats.confidence_interval.upper:.6f}]",
        f"  Margin of error: +/-{cv_stats.confidence_interval.margin_of_error:.6f}",
    ]

    # Append normality test results if available
    if cv_stats.normality_test is not None:
        nt = cv_stats.normality_test
        normal_str = "Yes" if nt.is_normal else "No"
        lines.extend(
            [
                "",
                f"Normality Test (Shapiro-Wilk, alpha={nt.alpha}):",
                f"  Statistic: {nt.statistic:.6f}",
                f"  p-value:   {nt.p_value:.6f}",
                f"  Normal:    {normal_str}",
            ]
        )

    return "\n".join(lines)


def format_model_comparison_report(comparison: ModelComparisonResult) -> str:
    """
    Format model comparison results as a human-readable text report.

    Generates a structured report showing mean scores, difference, and
    results from both parametric and non-parametric hypothesis tests.

    :param ModelComparisonResult comparison: ModelComparisonResult object to format.
    :return str: Multi-line formatted string report.
    """
    lines = [
        f"Model Comparison: {comparison.model_a_name} vs {comparison.model_b_name}",
        "=" * 60,
        "",
        "Mean Scores:",
        f"  {comparison.model_a_name}: {comparison.model_a_mean:.6f}",
        f"  {comparison.model_b_name}: {comparison.model_b_mean:.6f}",
        f"  Difference (A - B): {comparison.mean_difference:.6f}",
        "",
        f"Paired t-test (alpha={comparison.paired_ttest.alpha}):",
        f"  t-statistic: {comparison.paired_ttest.statistic:.6f}",
        f"  p-value:     {comparison.paired_ttest.p_value:.6f}",
        f"  Significant: {'Yes' if comparison.paired_ttest.is_significant else 'No'}",
    ]

    # Append effect size if available
    if comparison.paired_ttest.effect_size is not None:
        lines.append(
            f"  Cohen's d:   {comparison.paired_ttest.effect_size:.4f} "
            f"({comparison.paired_ttest.effect_size_interpretation})"
        )

    # Append Wilcoxon test results if available
    if comparison.wilcoxon_test is not None:
        wt = comparison.wilcoxon_test
        lines.extend(
            [
                "",
                f"Wilcoxon Signed-Rank Test (alpha={wt.alpha}):",
                f"  W-statistic: {wt.statistic:.6f}",
                f"  p-value:     {wt.p_value:.6f}",
                f"  Significant: {'Yes' if wt.is_significant else 'No'}",
            ]
        )

    return "\n".join(lines)
