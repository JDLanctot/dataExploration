import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from typing import Tuple, List, Union

__all__ = []
__all__.extend([
    'density_lookup',
    'add_per_accommodates',
    'examine_correlations',
    'remove_highly_correlated_features',
    'feature_importance_from_model'
])

def density_lookup(density: np.ndarray, lat_bins: np.ndarray, lon_bins: np.ndarray, lat: float, lon: float) -> float:
    lat_idx = np.digitize(lat, lat_bins) - 1
    lon_idx = np.digitize(lon, lon_bins) - 1
    # HANDLE PROBLEMS IF A VALUE FALLS EXACTLY ON THE END OF THE LAT OR LON LATTICE
    lat_idx = min(lat_idx, density.shape[0] - 1)
    lon_idx = min(lon_idx, density.shape[1] - 1)
    return density[lat_idx, lon_idx]

def add_per_accommodates(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    df[f'{colname}_per_accommodates'] = df[colname] / df['accommodates']
    # df = df.drop(colname, axis=1)
    return df

def examine_correlations(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    corr_matrix = df.corr()
    print(corr_matrix[colname].sort_values(ascending=False))
    print('-'*90)
    return corr_matrix

def remove_highly_correlated_features(df: pd.DataFrame, corr_matrix: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, List]:
    # FIND THE VERY CORRELATED COLUMNS BUT IGNORE THE DIAGONAL (SELF CORRELATION)
    upper = corr_matrix.abs().where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # DROP FEATURES AND RETURN WHICH ONES WERE DROPPED
    df_reduced = df.drop(columns=to_drop)
    return df_reduced, to_drop

def feature_importance_from_model(model: Union[LinearRegression, LassoCV, RandomForestRegressor, DecisionTreeRegressor], feature_names: List[str]) -> pd.DataFrame:
    # GETTING FEATURE IMPORTANCE FROM MODEL COEFFICIENTS
    # DETERMINE THE CORRECT OBJECT VARIABLE FOR IMPORTANCE BASED ON THE MODEL
    if hasattr(model, 'coef_'):
        # LINEAR AND LASSO
        importance_values = np.abs(model.coef_)
    elif hasattr(model, 'feature_importances_'):
        # TREE AND FORREST
        importance_values = model.feature_importances_
    else:
        raise ValueError("Model does not have recognized importance attribute.")

    importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance_values})
    importance = importance.sort_values(by='Importance', ascending=False)
    return importance

