import pandas as pd
import polars as pl

def calc_mb_usage(df: pd.DataFrame, deep:bool=True) -> float:
    return df.memory_usage(deep=deep).sum() / 1024 ** 2

def pl_calc_mb_usage(df: pl.LazyFrame) -> float:
    return df.estimated_size('mb')