import pandas as pd
import dhi.data.constants as dconst

from dhi.utils import get_logger
from dhi.decorators import time_func
from dhi.data.utils import reduce_memory_usage


class Preprocessor:
    def __init__(self, **kwargs) -> None:
        self._init_from_kwargs(**kwargs)

    def _init_from_kwargs(self, **kwargs) -> None:
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
        self.nan_category = kwargs.get("nan_category", dconst.DHI_PREPROCESSOR_NAN_CATEGORY)
        if not isinstance(self.nan_category, str):
            self.logger.warning(f"nan_category must be a string, defaulting to {dconst.DHI_PREPROCESSOR_NAN_CATEGORY}")
            self.nan_category = dconst.DHI_PREPROCESSOR_NAN_CATEGORY

        # Numerical to categorical features mapping
        self.numerical_to_categorical_features = kwargs.get("numerical_to_categorical_features", {})
        assert isinstance(
            self.numerical_to_categorical_features, dict
        ), "numerical_to_categorical_features must be a dictionary"
        assert all(
            isinstance(mapping, dict) for mapping in self.numerical_to_categorical_features.values()
        ), "all mapping values must be dictionaries"

        # Mapping {feature: scaler}
        self._numerical_scalers = {}
        self._categorical_encoders = {}

    @property
    def numerical_scalers(self) -> dict:
        return self._numerical_scalers

    @property
    def categorical_encoders(self) -> dict:
        return self._categorical_encoders

    def _get_transform_info(self, transform_info: object, default_type: str) -> tuple[str, dict]:
        if not transform_info:
            self.logger.warning(f"No transform information provided. Falling back to {default_type}")
            return default_type, {}

        if not isinstance(transform_info, dict):
            self.logger.warning(f"transform_info must be a dictionary, defaulting to {default_type}")
            return default_type, {}

        transform_type = transform_info.get("type", default_type)
        params = transform_info.get("params", {})

        if not isinstance(transform_type, str):
            self.logger.warning(f"transform_info['type'] must be a string, defaulting to {default_type}")
            transform_type = default_type

        if not isinstance(params, dict):
            self.logger.warning("transform_info['params'] must be a dictionary, defaulting to empty dictionary")
            params = {}

        return transform_type, params

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
                most_common_value = df_nan_handled[feature].mode(dropna=True)
                if most_common_value is None or most_common_value.empty:
                    self.logger.warning(f"No most common value found for feature {feature}, skipping NaN handling")
                    continue
                df_nan_handled[feature] = df_nan_handled[feature].fillna(most_common_value.iloc[0])

        return df_nan_handled

    @time_func
    def _handle_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Handles numerical features by scaling them according to the scaler provided in the config.

        If the scaler is not provided, a default one is used.

        If a feature to be transformed is not found in the columns of the DataFrame,
        this is reported and the feature is skipped.

        :param pd.DataFrame df: The DataFrame to handle the numerical features
        :param bool fit: Whether to fit the scaler or just transform, defaults to True
        :return pd.DataFrame: The DataFrame with the numerical features handled
        """
        if self.dry_run:
            self.logger.info("Dry run mode, skipping numerical feature handling")
            return df

        scaler_type, scaler_params = self._get_transform_info(
            transform_info=self.numerical_features.get("scaler", {}),
            default_type=dconst.DHI_PREPROCESSOR_DEFAULT_SCALER,
        )

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
            feature_values = df_numerical_transformed[feature_to_transform].to_numpy().reshape(-1, 1)
            if fit or feature_to_transform not in self.numerical_scalers:
                scaler_instance = dconst.DHI_PREPROCESSOR_NUMERICAL_SCALERS[scaler_type](
                    **scaler_params
                )  # pyright: ignore[reportCallIssue]
                fitted_scaler_instance = scaler_instance.fit(  # pyright: ignore[reportAttributeAccessIssue]
                    feature_values
                )  # pyright: ignore[reportAttributeAccessIssue]
                self.numerical_scalers[feature_to_transform] = fitted_scaler_instance
            else:
                fitted_scaler_instance = self.numerical_scalers[feature_to_transform]

            df_numerical_transformed[feature_to_transform] = fitted_scaler_instance.transform(feature_values).flatten()

        return df_numerical_transformed

    @time_func
    def _handle_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Handles categorical features by encoding them according to the encoder provided in the config.

        If the encoder is not provided, a default one is used.

        If a feature to be encoded is not found in the columns of the DataFrame,
        this is reported and the feature is skipped.

        :param pd.DataFrame df: The DataFrame to handle the categorical features
        :param bool fit: Whether to fit the encoder or just transform, defaults to True
        :return pd.DataFrame: The DataFrame with the categorical features handled
        """
        if self.dry_run:
            self.logger.info("Dry run mode, skipping categorical feature handling")
            return df

        encoder_type, encoder_params = self._get_transform_info(
            transform_info=self.categorical_features.get("encoder", {}),
            default_type=dconst.DHI_PREPROCESSOR_DEFAULT_ENCODER,
        )

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
            original_values = df_categorical_encoded[feature_to_encode].to_numpy()
            values = original_values
            # LabelEncoder is the only encoder that requires a 1D array
            if encoder_type != "label_encoder":
                values = values.reshape(-1, 1)
            if fit or feature_to_encode not in self.categorical_encoders:
                encoder_instance = dconst.DHI_PREPROCESSOR_CATEGORICAL_ENCODERS[encoder_type](
                    **encoder_params
                )  # pyright: ignore[reportCallIssue]
                fitted_encoder_instance = encoder_instance.fit(values)  # pyright: ignore[reportAttributeAccessIssue]
                self.categorical_encoders[feature_to_encode] = fitted_encoder_instance
            else:
                fitted_encoder_instance = self.categorical_encoders[feature_to_encode]

            # If using LabelEncoder, unseen categories in transform() will throw an error; replace unseen with nan_category
            if encoder_type == "label_encoder" and hasattr(fitted_encoder_instance, "classes_"):
                known = set(getattr(fitted_encoder_instance, "classes_"))
                if self.nan_category in known:
                    values_1d = original_values
                    values_1d = (
                        pd.Series(values_1d).where(pd.Series(values_1d).isin(known), self.nan_category).to_numpy()
                    )
                    values = values_1d

            df_categorical_encoded[feature_to_encode] = fitted_encoder_instance.transform(values)

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
            features = list(self.numerical_scalers.keys() | self.categorical_encoders.keys())
        elif isinstance(features, str):
            features = [features]
        elif isinstance(features, list) and all(isinstance(feature, str) for feature in features):
            features = list(set(features))
        else:
            raise ValueError("features must be a string, list of strings, or None, got {type(features)}")

        df_inverse_transformed = df.copy()

        for feature in features:
            if feature in self.numerical_scalers:
                scaler_instance = self.numerical_scalers[feature]
            elif feature in self.categorical_encoders:
                scaler_instance = self.categorical_encoders[feature]
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

    def _apply_preprocessing_steps(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
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
        df = self._handle_numerical_features(df, fit=fit)
        df = self._handle_categorical_features(df, fit=fit)

        if not self.dry_run:
            df = reduce_memory_usage(df, logger=self.logger)

        return df

    @time_func
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the preprocessor to the DataFrame and transforms it.

        :param pd.DataFrame df: The DataFrame to fit and transform
        :return pd.DataFrame: The transformed DataFrame
        """
        return self._apply_preprocessing_steps(df, fit=True)

    @time_func
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame using the already fitted preprocessor.

        :param pd.DataFrame df: The DataFrame to transform
        :return pd.DataFrame: The transformed DataFrame
        """
        if not self.numerical_scalers and not self.categorical_encoders:
            self.logger.warning("Preprocessor instance has not been fitted yet. Proceeding with fit during transform.")
            return self._apply_preprocessing_steps(df, fit=True)
        return self._apply_preprocessing_steps(df, fit=False)
