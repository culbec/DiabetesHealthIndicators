import pandas as pd
import sklearn.preprocessing as skp

import dhi.data.preprocessor.const as dppconst
from dhi.data.loader._base import reduce_memory_usage
from dhi.decorators import time_func
from dhi.utils import get_logger


class Preprocessor(object):
    def __init__(self, **kwargs) -> None:
        self.init(**kwargs)

    def init(self, **kwargs) -> None:
        self.logger = get_logger(self.__class__.__name__)

        # Numerical features
        self.numerical_features = kwargs.get("numerical_features", {})
        assert isinstance(self.numerical_features, dict), "numerical_features must be a dictionary"

        # Categorical features
        self.categorical_features = kwargs.get("categorical_features", {})
        assert isinstance(self.categorical_features, dict), "categorical_features must be a dictionary"

        # Controls if the preprocessor should run or just print the steps
        self.dry_run = kwargs.get("dry_run", False)
        assert isinstance(self.dry_run, bool), "dry_run must be a boolean"

        # Nan category
        self.nan_category = kwargs.get("nan_category", dppconst.DHI_PREPROCESSOR_NAN_CATEGORY)
        if not isinstance(self.nan_category, str):
            self.logger.warning(
                f"nan_category must be a string, defaulting to {dppconst.DHI_PREPROCESSOR_NAN_CATEGORY}"
            )
            self.nan_category = dppconst.DHI_PREPROCESSOR_NAN_CATEGORY

        # Numerical to categorical features mapping
        self.numerical_to_categorical_features = kwargs.get("numerical_to_categorical_features", {})
        assert isinstance(
            self.numerical_to_categorical_features, dict
        ), "numerical_to_categorical_features must be a dictionary"
        assert all(
            isinstance(mapping, dict) for mapping in self.numerical_to_categorical_features.values()
        ), "all mapping values must be dictionaries"

        # mapping {feature: scaler}
        self._numerical_scaling_data = {}
        self._categorical_encoding_data = {}

    @property
    def numerical_scaling_data(self) -> dict:
        return self._numerical_scaling_data

    @property
    def categorical_encoding_data(self) -> dict:
        return self._categorical_encoding_data

    @time_func
    def transform_numerical_to_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms numerical features to categorical features according to
        the mapping provided in the config.

        If the mapping is not provided, or if a feature name does not appear in the mapping,
        the feature is transformed to a string.

        :param pd.DataFrame df: The DataFrame to transform the numerical features to categorical features
        :return pd.DataFrame: The DataFrame with the numerical features transformed to categorical features
        """
        if self.dry_run:
            self.logger.info("Dry run mode, skipping numerical to categorical transformation")
            return df

        for feature, mapping in self.numerical_to_categorical_features.items():
            if feature not in df.columns:
                self.logger.warning(f"Feature {feature} not found in DataFrame")
                continue

            # Convert column to object dtype first to avoid FutureWarning
            df[feature] = df[feature].astype("object")

            for value in df[feature].unique():
                if value not in mapping:
                    df.loc[df[feature] == value, feature] = str(value)
                else:
                    df.loc[df[feature] == value, feature] = mapping[value]

        return df

    @time_func
    def _handle_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles NaN values by:

        * Numerical features: linear interpolation
        * Categorical features: replacing with the nan_category
        * Other features: most common value

        :param pd.DataFrame df: The DataFrame to handle the NaN values
        :return pd.DataFrame: The DataFrame with the NaN values handled
        """
        if self.dry_run:
            self.logger.info("Dry run mode, skipping NaN handling")
            return df

        df_nan_handled = df.copy()

        for feature in df_nan_handled.columns:
            if pd.api.types.is_numeric_dtype(df_nan_handled[feature]):
                df_nan_handled[feature] = df_nan_handled[feature].interpolate(method="linear", limit_direction="both")
            elif pd.api.types.is_object_dtype(df_nan_handled[feature]):
                df_nan_handled[feature] = df_nan_handled[feature].fillna(self.nan_category)
            else:
                df_nan_handled[feature] = df_nan_handled[feature].fillna(df_nan_handled[feature].mode(dropna=True)[0])

        return df_nan_handled

    @time_func
    def _handle_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles numerical features by scaling them according to the scaler provided in the config.

        If the scaler is not provided, a default one is used.

        If a feature to be transformed is not found in the columns of the DataFrame,
        this is reported and the feature is skipped.

        :param pd.DataFrame df: The DataFrame to handle the numerical features
        :return pd.DataFrame: The DataFrame with the numerical features handled
        """
        if self.dry_run:
            self.logger.info("Dry run mode, skipping numerical feature handling")
            return df

        scaler_info = self.numerical_features.get("scaler", {})

        if not scaler_info:
            self.logger.warning(
                f"No scaler information provided. Falling back to {dppconst.DHI_PREPROCESSOR_DEFAULT_SCALER}"
            )
            scaler_info = {
                "type": dppconst.DHI_PREPROCESSOR_DEFAULT_SCALER,
                "params": {},
            }
        if not isinstance(scaler_info, dict):
            self.logger.warning(
                f"scaler_info must be a dictionary, defaulting to {dppconst.DHI_PREPROCESSOR_DEFAULT_SCALER}"
            )
            scaler_info = {
                "type": dppconst.DHI_PREPROCESSOR_DEFAULT_SCALER,
                "params": {},
            }
        if not isinstance(scaler_info.get("type", dppconst.DHI_PREPROCESSOR_DEFAULT_SCALER), str):
            self.logger.warning(
                f"scaler_info['type'] must be a string, defaulting to {dppconst.DHI_PREPROCESSOR_DEFAULT_SCALER}"
            )
            scaler_info["type"] = dppconst.DHI_PREPROCESSOR_DEFAULT_SCALER
        if not isinstance(scaler_info.get("params", {}), dict):
            self.logger.warning(f"scaler_info['params'] must be a dictionary, defaulting to {{}}")
            scaler_info["params"] = {}

        scaler_type = scaler_info.get("type", dppconst.DHI_PREPROCESSOR_DEFAULT_SCALER)
        scaler_params = scaler_info.get("params", {})

        features_to_transform = self.numerical_features.get("features", [])
        assert isinstance(features_to_transform, list), "features_to_transform must be a list"
        assert all(
            isinstance(feature, str) for feature in features_to_transform
        ), "all features_to_transform must be strings"

        numerical_dtype_features = df.select_dtypes(include=["number"]).columns.tolist()
        non_compliant_features = [
            feature for feature in features_to_transform if feature not in numerical_dtype_features
        ]

        if non_compliant_features:
            self.logger.warning(
                f"The following features are not numerical and will be skipped: {non_compliant_features}"
            )
            features_to_transform = [
                feature for feature in features_to_transform if feature not in non_compliant_features
            ]

        if not features_to_transform:
            self.logger.warning("No features to transform")
            return df

        df_numerical_transformed = df.copy()

        for feature_to_transform in features_to_transform:
            scaler_instance = dppconst.DHI_PREPROCESSOR_NUMERICAL_SCALERS[scaler_type](**scaler_params)
            fitted_scaler_instance = scaler_instance.fit(
                df_numerical_transformed[feature_to_transform].values.reshape(-1, 1)
            )
            df_numerical_transformed[feature_to_transform] = fitted_scaler_instance.transform(
                df_numerical_transformed[feature_to_transform].values.reshape(-1, 1)
            ).flatten()

            self.numerical_scaling_data[feature_to_transform] = fitted_scaler_instance

        return df_numerical_transformed

    @time_func
    def _handle_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles categorical features by encoding them according to the encoder provided in the config.

        If the encoder is not provided, a default one is used.

        If a feature to be encoded is not found in the columns of the DataFrame,
        this is reported and the feature is skipped.
        """
        if self.dry_run:
            self.logger.info("Dry run mode, skipping categorical feature handling")
            return df

        encoder_info = self.categorical_features.get("encoder", {})
        if not encoder_info:
            self.logger.warning(
                f"No encoder information provided. Falling back to {dppconst.DHI_PREPROCESSOR_DEFAULT_ENCODER}"
            )
            encoder_info = {
                "type": dppconst.DHI_PREPROCESSOR_DEFAULT_ENCODER,
                "params": {},
            }
        if not isinstance(encoder_info, dict):
            self.logger.warning(
                f"encoder_info must be a dictionary, defaulting to {dppconst.DHI_PREPROCESSOR_DEFAULT_ENCODER}"
            )
            encoder_info = {
                "type": dppconst.DHI_PREPROCESSOR_DEFAULT_ENCODER,
                "params": {},
            }
        if not isinstance(encoder_info.get("type", dppconst.DHI_PREPROCESSOR_DEFAULT_ENCODER), str):
            self.logger.warning(
                f"encoder_info['type'] must be a string, defaulting to {dppconst.DHI_PREPROCESSOR_DEFAULT_ENCODER}"
            )
            encoder_info["type"] = dppconst.DHI_PREPROCESSOR_DEFAULT_ENCODER
        if not isinstance(encoder_info.get("params", {}), dict):
            self.logger.warning(f"encoder_info['params'] must be a dictionary, defaulting to {{}}")
            encoder_info["params"] = {}

        encoder_type = encoder_info.get("type", dppconst.DHI_PREPROCESSOR_DEFAULT_ENCODER)
        encoder_params = encoder_info.get("params", {})

        features_to_encode = self.categorical_features.get("features", [])
        assert isinstance(features_to_encode, list), "features_to_encode must be a list"
        assert all(
            isinstance(feature, str) for feature in features_to_encode
        ), "all features_to_encode must be strings"

        object_dtype_features = df.select_dtypes(
            include=["object", "string", "category", "datetime", "datetime64"]
        ).columns.tolist()
        non_compliant_features = [feature for feature in features_to_encode if feature not in object_dtype_features]

        if non_compliant_features:
            self.logger.warning(
                f"The following features are not categorical and will be skipped: {non_compliant_features}"
            )
            features_to_encode = [feature for feature in features_to_encode if feature not in non_compliant_features]

        if not features_to_encode:
            self.logger.warning("No features to encode")
            return df

        df_categorical_encoded = df.copy()

        for feature_to_encode in features_to_encode:
            encoder_instance = dppconst.DHI_PREPROCESSOR_CATEGORICAL_ENCODERS[encoder_type](**encoder_params)
            fitted_encoder_instance = encoder_instance.fit(df_categorical_encoded[feature_to_encode].values)
            df_categorical_encoded[feature_to_encode] = fitted_encoder_instance.transform(
                df_categorical_encoded[feature_to_encode].values
            )

            self.categorical_encoding_data[feature_to_encode] = fitted_encoder_instance

        return df_categorical_encoded

    @time_func
    def feature_inverse_transform(self, df: pd.DataFrame, features: str | list[str] | None = None) -> pd.DataFrame:
        """
        Inverse transforms the feature/features/all_features back to their original values.

        :param pd.DataFrame df: The DataFrame to inverse transform the feature/features/all_features back to their original values
        :param str | list[str] | None features: The feature/features/all_features to inverse transform back to their original values, defaults to None
        :return pd.DataFrame: The DataFrame with the feature/features/all_features inverse transformed back to their original values
        :raises ValueError: If the features are not a string, list of strings, or None
        """
        if not features:
            self.logger.warning("No features provided, inverse transforming all features")
            features = self.numerical_scaling_data.keys() | self.categorical_encoding_data.keys()
        elif isinstance(features, str):
            features = [features]
        elif isinstance(features, list) and all(isinstance(feature, str) for feature in features):
            features = list(set(features))
        else:
            raise ValueError("features must be a string, list of strings, or None, got {type(features)}")

        df_inverse_transformed = df.copy()

        for feature in features:
            if feature in self.numerical_scaling_data:
                scaler_instance = self.numerical_scaling_data[feature]
            elif feature in self.categorical_encoding_data:
                scaler_instance = self.categorical_encoding_data[feature]
            else:
                self.logger.warning(
                    f"Feature {feature} not found in the numerical scaling or categorical encoding data"
                )
                continue

            if not getattr(scaler_instance, "inverse_transform", None):
                self.logger.warning(
                    f"Scaler/Encoder {scaler_instance.__class__.__name__} has no inverse transform method"
                )
                continue

            df_inverse_transformed[feature] = scaler_instance.inverse_transform(df_inverse_transformed[feature])

        return df_inverse_transformed

    @time_func
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the DataFrame by:

        * Handling NaN values
        * Transforming numerical features to categorical features
        * Scaling numerical features
        * Encoding categorical features
        * Reduces the memory usage of the DataFrame in the end
        """

        df = self._handle_nan(df)
        df = self.transform_numerical_to_categorical(df)
        df = self._handle_numerical_features(df)
        df = self._handle_categorical_features(df)

        if not self.dry_run:
            df = reduce_memory_usage(df, logger=self.logger)

        return df
