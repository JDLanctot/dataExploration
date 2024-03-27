import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from typing import Tuple, List, Union

__all__ = []
__all__.extend([
    'examine_correlations',
    'remove_highly_correlated_features',
    'feature_importance_from_model'
])

def examine_correlations(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    corr_matrix = df.corr()
    print(corr_matrix[colname].sort_values(ascending=False))
    return corr_matrix

def remove_highly_correlated_features(df: pd.DataFrame, corr_matrix: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, List]:
    # FIND THE VERY CORRELATED COLUMNS BUT IGNORE THE DIAGONAL (SELF CORRELATION)
    upper = corr_matrix.abs().where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # DROP FEATURES AND RETURN WHICH ONES WERE DROPPED
    df_reduced = df.drop(columns=to_drop)
    return df_reduced, to_drop

def feature_importance_from_model(model: Union[LinearRegression, LassoCV], feature_names: List[str]) -> pd.DataFrame:
    # GETTING FEATURE IMPORTANCE FROM MODEL COEFFICIENTS
    importance = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(model.coef_)})
    importance = importance.sort_values(by='Importance', ascending=False)

    return importance

