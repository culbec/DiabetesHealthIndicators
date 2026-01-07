import shap
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from numpy.typing import ArrayLike

import dhi.constants as dconst
from dhi.utils import get_logger
from dhi.decorators import time_func
from dhi.models.experiments.runner import ExperimentRunner


class SHAPModelExplainer:
    def __init__(self, runner: ExperimentRunner) -> None:
        self.logger = get_logger(self.__class__.__name__)

        self._runner = runner

        self.explainer_config = dconst.DHI_ML_EXPLAINER_REGISTRY.get(runner.model_name)
        assert self.explainer_config is not None, f"No explainer config found for model {runner.model_name}"

    @time_func
    def compute_shap_values(
        self,
        X: ArrayLike,
        feature_names: list[str],
        with_plots: bool = True,
        background_samples: int | None = 100,
        top_k_dependence: int = 10,
        with_interactions: bool = False,
    ) -> dconst.ShapExplanationsType:
        """
        Computes the SHAP values and interactions for the given model underneath the runner hood.

        :param ArrayLike X: The input data
        :param list[str] feature_names: The feature names
        :param bool with_plots: Whether to plot the SHAP values, defaults to True
        :param int | None background_samples: Number of background samples for KernelExplainer (uses kmeans clustering).
        None means all samples are used. Defaults to 100.
        :param int top_k_dependence: Number of top features (by mean SHAP) to use for dependence plots. Defaults to 10
        :param bool with_interactions: Whether to compute and plot SHAP interaction values. Defaults to False.
        :return ShapExplanationsType: The SHAP values and interactions
        """
        self.logger.info(
            f"Computing SHAP values for model {self._runner.model_name} with data of shape: X {X.shape} and feature names: {feature_names}"
        )

        # KernelExplainer expects a callable (e.g., model.predict), not the model object itself
        # Explain probability for the positive class for classification tasks or the predicted value for regression
        task_type = getattr(self._runner, "task_type", None)
        use_proba = task_type == "classification" and callable(getattr(self._runner.model, "predict_proba", None))

        if self.explainer_config["class"] == shap.explainers.KernelExplainer:
            if use_proba:
                model_arg = lambda X_in: self._runner.model.predict_proba(X_in)[:, 1]
            else:
                model_arg = self._runner.model.predict
            if background_samples is not None and X.shape[0] > background_samples:
                self.logger.info(
                    f"Summarizing background data from {X.shape[0]} to {background_samples} samples using kmeans"
                )
                # Summarize background data using kmeans for better performance
                background_data = shap.kmeans(X, background_samples)
            else:
                background_data = X
        else:
            # Other explainers can accept the model directly
            model_arg = self._runner.model
            background_data = X

        explainer = self.explainer_config["class"](
            model_arg, data=background_data, feature_names=feature_names, **self.explainer_config["kwargs"]
        )

        expected_value = explainer.expected_value
        explanation = explainer(X)

        shap_values = np.asarray(explanation.values)
        # Reduce SHAP values to 2D for ranking or plotting
        shap_matrix = shap_values
        if shap_values.ndim == 3:
            if shap_matrix.shape[1] == 2:
                shap_matrix = shap_matrix[:, 1, :]  # Prefer positive class explanation for binary classification
            else:
                shap_matrix = np.mean(shap_matrix, axis=1)  # Average over classes for multi-class classification

        self.logger.info(
            f"Computing SHAP interactions for model {self._runner.model_name} with data of shape: X {X.shape} and feature names: {feature_names}"
        )
        shap_interaction_values = None
        if with_interactions:
            if not hasattr(explainer, "shap_interaction_values"):
                self.logger.warning(f"Model {self._runner.model_name} does not support SHAP interactions")
            else:
                shap_interaction_values = np.asarray(explainer.shap_interaction_values(X))

        # TODO: separate plotting to methods?
        if with_plots:
            # NOTE: closing all figures of matplotlib because plotyl and matplotlib are mixing
            plt.clf()
            plt.close("all")
            self.logger.info(f"Generating SHAP plots...")

            # Beeswarm plot - global model explanation
            self.logger.info(f"Beeswarm plot - global model explanation")
            shap.plots.beeswarm(explanation)

            # Bar plot - global model explanation
            self.logger.info(f"Bar plot - global model explanation - mean feature contribution")
            shap.plots.bar(explanation)

            # Decision plot - how did a model arrive to a decision?
            n_samples = int(min(dconst.DHI_SHAP_SUBSAMPLE_SIZE, X.shape[0]))
            shap.decision_plot(
                expected_value,
                shap_matrix[:n_samples, :],
                features=X[:n_samples],
                feature_names=feature_names,
                link="logit" if use_proba else "identity",
                ignore_warnings=True,
            )

            # Waterfall plot - local analysis
            self.logger.info(f"Waterfall plot - local analysis")
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
                        features=X,
                        feature_names=feature_names,
                    )

            if with_interactions and shap_interaction_values is not None:
                self.logger.info(f"Generating SHAP interaction plots...")

                shap.summary_plot(shap_interaction_values, X)

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
