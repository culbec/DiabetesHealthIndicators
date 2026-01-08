import pandas as pd

from dhi.data.preprocessing._base import Preprocessor


class DHIDataPreprocessor(Preprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # NOTE: add any extra preprocessing behavior here if needed
        return super().fit_transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # NOTE: add any extra preprocessing behavior here if needed
        return super().transform(df)
