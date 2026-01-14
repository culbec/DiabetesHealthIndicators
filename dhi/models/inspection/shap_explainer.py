from typing import Optional, Tuple, TypeAlias

import matplotlib.pyplot as plt
import shap
import numpy as np
import plotly.graph_objects as go

from shap import Explanation
from numpy.typing import ArrayLike
from shap.maskers import Independent

import dhi.constants as dconst
from dhi.decorators import time_func
from dhi.models.experiments.runner import ExperimentRunner
from dhi.utils import get_logger

ShapExplanationsType: TypeAlias = Tuple[Optional[ArrayLike], Optional[ArrayLike]]


class SHAPModelExplainer:
    def __init__(self, runner: ExperimentRunner) -> None:
        self.logger = get_logger(self.__class__.__name__)

        self._runner = runner

        self._explainer_config = dconst.DHI_ML_EXPLAINER_REGISTRY.get(self._runner.model_name, {})
        assert self._explainer_config is not None, f"No explainer config found for model {self._runner.model_name}"

    @time_func
    def compute_shap_values(
        self,
        X_train: ArrayLike,
        X_test: ArrayLike,
        feature_names: list[str],
        with_plots: bool = True,
        background_samples: int = dconst.DHI_SHAP_BACKGROUND_SAMPLES,
        top_k_dependence: int = 10,
        with_interactions: bool = False,
    ) -> ShapExplanationsType:
        """
        Computes the SHAP values and interactions for the given model underneath the runner hood.

        :param ArrayLike X_train: The training input data
        :param ArrayLike X_test: The test input data
        :param list[str] feature_names: The feature names
        :param bool with_plots: Whether to plot the SHAP values, defaults to True
        :param int | None background_samples: Number of background samples for Explainer (uses kmeans clustering).
        None means all samples are used. Defaults to dconst.DHI_SHAP_BACKGROUND_SAMPLES.
        :param int top_k_dependence: Number of top features (by mean absolute SHAP) to use for dependence plots. Defaults to 10
        :param bool with_interactions: Whether to compute and plot SHAP interaction values. Defaults to False.
        :return ShapExplanationsType: The SHAP values and interactions
        """
        X_train, X_test = np.asarray(X_train), np.asarray(X_test)

        self.logger.info(
            f"Computing SHAP values for model {self._runner._model_name} with training data of shape: X_train {X_train.shape}, test data of shape: X_test {X_test.shape} and feature names: {feature_names}"
        )

        # Explainer expects a callable (e.g., model.predict), not the model object itself
        # Explain probability for the positive class for classification tasks or the predicted value for regression
        task_type = getattr(self._runner, "task_type", None)
        use_proba = task_type == "classification" and callable(getattr(self._runner._model, "predict_proba", None))

        def model_arg(X_in):
            pred_proba = getattr(self._runner._model, "predict_proba", None)
            if use_proba and callable(pred_proba):
                proba: np.ndarray = pred_proba(X_in)  # pyright: ignore[reportAssignmentType]
                # If output is 1d (binary), just return it; if 2d, select positive class column
                if proba.ndim == 1:
                    return proba
                elif proba.ndim == 2 and proba.shape[1] > 1:
                    return proba[:, 1]
                else:
                    # Could be shape (n_samples, 1)
                    return proba[:, 0]
            else:
                return self._runner._model.predict(X_in)

        # Up-casting the input data, if float16 is passed
        # shap doesn't do this automatically, and the code fails with NotImplementedError
        X_train = np.asarray(X_train)
        if X_train.dtype == np.float16:
            X_train = X_train.astype(np.float32)
        X_test = np.asarray(X_test)
        if X_test.dtype == np.float16:
            X_test = X_test.astype(np.float32)

        # Using an Independent masker in order to mask the samples
        # which will be used to explain the reasoning of the model
        # Sub-samples the data, not clustering common points as KMeans does
        masker = Independent(data=X_train, max_samples=background_samples)

        explainer = self._explainer_config["class"](
            self._runner._model,
            # masker=masker,
            data=X_train,
            feature_names=feature_names,
            **self._explainer_config["kwargs"],
        )

        expected_value = explainer.expected_value if hasattr(explainer, "expected_value") else None
        explanation = explainer(X_test)

        shap_values = np.asarray(explanation.values)
        # Reduce SHAP values to 2D for ranking or plotting
        shap_matrix = shap_values
        if shap_values.ndim == 3:
            if shap_matrix.shape[2] == 2:
                shap_matrix = shap_matrix[:, :, 1]  # Prefer positive class explanation for binary classification
            else:
                shap_matrix = np.mean(shap_matrix, axis=2)  # Average over classes for multi-class classification

        base_vals = explanation.base_values
        if isinstance(base_vals, np.ndarray) and base_vals.ndim > 1:
            base_vals = base_vals[:, 1] if base_vals.shape[1] == 2 else base_vals.mean(axis=1)

        explanation = Explanation(
            values=shap_matrix, base_values=base_vals, data=explanation.data, feature_names=explanation.feature_names
        )

        self.logger.info(
            f"Computing SHAP interactions for model {self._runner._model_name} with data of shape: X_test {X_test.shape} and feature names: {feature_names}"
        )
        shap_interaction_values = None
        if with_interactions:
            if not hasattr(explainer, "shap_interaction_values"):
                self.logger.warning(f"Model {self._runner._model_name} does not support SHAP interactions")
            else:
                shap_interaction_values = np.asarray(explainer.shap_interaction_values(X_test))

        if with_plots:
            # NOTE: closing all figures of matplotlib because plotly and matplotlib are mixing
            plt.clf()
            plt.close("all")
            self.logger.info("Generating SHAP plots")

            # Beeswarm plot - global model explanation
            self.logger.info("Beeswarm plot - global model explanation")
            shap.plots.beeswarm(explanation)

            # Bar plot - global model explanation
            self.logger.info("Bar plot - global model explanation - mean feature contribution")
            shap.plots.bar(explanation)

            # Decision plot - how did a model arrive to a decision?
            if expected_value is not None:
                if isinstance(expected_value, np.ndarray) and expected_value.ndim > 0:
                    if len(expected_value) == 2:
                        expected_value = expected_value[1]  # Positive class for binary classification
                    elif len(expected_value) > 2:
                        expected_value = expected_value.mean()  # Average for multi-class

                n_samples = int(min(dconst.DHI_SHAP_SUBSAMPLE_SIZE, X_test.shape[0]))
                shap.decision_plot(
                    expected_value,
                    shap_matrix[:n_samples, :],
                    features=X_test[:n_samples],
                    feature_names=feature_names,
                    link="logit" if use_proba else "identity",
                    ignore_warnings=True,
                )

            # Waterfall plot - local analysis
            self.logger.info("Waterfall plot - local analysis")
            shap.plots.waterfall(explanation[0], max_display=dconst.DHI_SHAP_WATERFALL_MAX_DISPLAY)

            # Dependence plot - local analysis
            self.logger.info(f"Dependence plot - local analysis (top {top_k_dependence})")
            mean_abs = np.mean(np.abs(shap_matrix), axis=0)
            k = int(max(0, min(top_k_dependence, len(feature_names))))
            if k > 0:
                top_idx = np.argsort(mean_abs)[::-1][:k]
                for j in top_idx:
                    feature_name = feature_names[j]
                    shap.dependence_plot(
                        ind=feature_name,
                        shap_values=shap_matrix,
                        features=X_test,
                        feature_names=feature_names,
                    )

            if with_interactions and shap_interaction_values is not None:
                self.logger.info("Generating SHAP interaction plots")

                shap.summary_plot(shap_interaction_values, X_test)

                # Interaction analysis plot
                self.logger.info(
                    f"Interaction values: {shap_interaction_values.shape} | Max: {np.max(shap_interaction_values)} | Min: {np.min(shap_interaction_values)} | Mean: {np.mean(shap_interaction_values)} | Std: {np.std(shap_interaction_values)}"
                )

                interaction_strength = np.abs(shap_interaction_values).mean(axis=0)
                interaction_strength /= interaction_strength.max()

                fig = go.Figure(
                    data=go.Heatmap(
                        z=interaction_strength,
                        x=feature_names,
                        y=feature_names,
                        colorbar={"title": "Interaction Strength"},
                    )
                )

                fig.update_layout(
                    title="SHAP Interaction Analysis",
                    xaxis_title="Features",
                    yaxis_title="Features",
                    width=dconst.DHI_PLOT_WIDTH,
                    height=dconst.DHI_PLOT_HEIGHT,
                )

                fig.show()

            self.logger.info("SHAP plotting completed successfully")

        return shap_values, shap_interaction_values
