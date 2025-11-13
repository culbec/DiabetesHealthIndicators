import sklearn.preprocessing as skp
from sklearn.base import BaseEstimator

##############################
# PREPROCESSOR CONSTANTS
##############################

DHI_PREPROCESSOR_NUMERICAL_SCALERS: dict[str, BaseEstimator] = {
    "standard_scaler": skp.StandardScaler,
    "min_max_scaler": skp.MinMaxScaler,
    "max_abs_scaler": skp.MaxAbsScaler,
    "robust_scaler": skp.RobustScaler,
    "power_transformer": skp.PowerTransformer,
    "quantile_transformer": skp.QuantileTransformer,
}

DHI_PREPROCESSOR_CATEGORICAL_ENCODERS: dict[str, BaseEstimator] = {
    "label_encoder": skp.LabelEncoder,
    "one_hot_encoder": skp.OneHotEncoder,
    "label_binarizer": skp.LabelBinarizer,
    "multi_label_binarizer": skp.MultiLabelBinarizer,
}

DHI_PREPROCESSOR_DEFAULT_SCALER: str = "robust_scaler"
DHI_PREPROCESSOR_DEFAULT_ENCODER: str = "label_encoder"

DHI_PREPROCESSOR_NAN_CATEGORY: str = "Unknown"
DHI_PREPROCESSOR_DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"
