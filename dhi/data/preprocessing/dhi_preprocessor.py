import pandas as pd

from dhi.data.preprocessing._base import Preprocessor


class DHIPreprocessor(Preprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().preprocess(df)
        return df
