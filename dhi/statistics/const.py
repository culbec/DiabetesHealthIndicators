from typing import Any

##############################
# FEATURE SELECTION CONSTANTS
##############################

DHI_FEATURE_SELECTION_MODES: dict[str, dict[str, Any]] = {
    "percentile": {
        "param": 20,
    },
    "kbest": {
        "param": 10,
    },
    "fpr": {
        "param": 0.05,
    },
    "fdr": {
        "param": 0.05,
    },
    "fwe": {
        "param": 0.05,
    },
}

DHI_FEATURE_SELECTION_DEFAULT_MODE: str = "percentile"
DHI_FEATURE_SELECTION_DEFAULT_RELIEF_N_FEATURES: int = 10
DHI_FEATURE_SELECTION_DEFAULT_KBINS_N_BINS: int = 10

##############################
# COMPONENT REDUCTION CONSTANTS
##############################

DHI_COMPONENT_REDUCTION_DEFAULT_N_COMPONENTS: int = 2
