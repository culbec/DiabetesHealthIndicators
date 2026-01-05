import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import shap
from numpy.typing import ArrayLike

import dhi.constants as dconst
from dhi.decorators import time_func
from dhi.models.experiments.runner import ExperimentRunner
from dhi.utils import get_logger


class Explainer:
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
    ) -> dconst.ShapExplanationsType:
        """
        Computes the SHAP values and interactions for the given model underneath the runner hood.

        :param ArrayLike X: The input data
        :param list[str] feature_names: The feature names
        :param bool with_plots: Whether to plot the SHAP values, defaults to True
        :param int | None background_samples: Number of background samples for KernelExplainer (uses kmeans clustering).
            Set to None to use all samples. Defaults to 100.
        :return dconst.ShapExplanationsType: The SHAP values and interactions
        """
        self.logger.info(
            f"Computing SHAP values for model {self._runner.model_name} with data of shape: X {X.shape} and feature names: {feature_names}"
        )

        # KernelExplainer expects a callable (e.g., model.predict), not the model object itself
        # Also summarize background data using kmeans for better performance
        if self.explainer_config["class"] == shap.explainers.KernelExplainer:
            model_arg = self._runner.model.predict
            if background_samples is not None and X.shape[0] > background_samples:
                self.logger.info(
                    f"Summarizing background data from {X.shape[0]} to {background_samples} samples using kmeans"
                )
                background_data = shap.kmeans(X, background_samples)
            else:
                background_data = X
        else:
            model_arg = self._runner.model
            background_data = X

        explainer = self.explainer_config["class"](
            model_arg, data=background_data, feature_names=feature_names, **self.explainer_config["kwargs"]
        )

        expected_value = explainer.expected_value
        explanation = explainer(X)

        shap_values = np.asarray(explanation.values)

        self.logger.info(
            f"Computing SHAP interactions for model {self._runner.model_name} with data of shape: X {X.shape} and feature names: {feature_names}"
        )
        shap_interaction_values = None
        if not hasattr(explainer, "shap_interaction_values"):
            self.logger.warning(f"Model {self._runner.model_name} does not support SHAP interactions")
        else:
            shap_interaction_values = np.asarray(explainer.shap_interaction_values(X))

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
            shap.decision_plot(
                expected_value,
                shap_values[: dconst.DHI_SHAP_SUBSAMPLE_SIZE, :],
                features=X,
                feature_names=feature_names,
                link="logit",
                ignore_warnings=True,
            )

            # Waterfall plot - local analysis
            self.logger.info(f"Waterfall plot - local analysis")
            shap.plots.waterfall(explanation[0], max_display=dconst.DHI_SHAP_WATERFALL_MAX_DISPLAY)

            # Dependence plot - local analysis
            self.logger.info(f"Dependence plot - local analysis")
            for feature_name in feature_names:
                shap.dependence_plot(
                    ind=feature_name,
                    shap_values=shap_values,
                    features=X,
                    feature_names=feature_names,
                )

            if shap_interaction_values is not None:
                self.logger.info(f"Generating SHAP interaction plots...")

                shap.summary_plot(shap_interaction_values, X)

                # Interaction anaylsis plot
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

            self.logger.info("Plotting of SHAP interaction analysis completed successfully")

        return shap_values, shap_interaction_values
