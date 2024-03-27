import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
from typing import Tuple, Union

__all__ = []
__all__.extend([
    'create_train_test_sets',
    'train_linear_model',
    'train_lasso_model',
    'evaluate_model',
    'run_train'
])

def create_train_test_sets(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 432) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_set, test_set

def train_linear_model(train_set: pd.DataFrame, colname: str) -> LinearRegression:
    train_X = train_set.drop(colname, axis=1)
    train_y = train_set[colname]

    lin_reg = LinearRegression()
    lin_reg.fit(train_X, train_y)

    return lin_reg

def train_lasso_model(train_set: pd.DataFrame, colname: str) -> LassoCV:
    train_X = train_set.drop(colname, axis=1)
    train_y = train_set[colname]

    lasso = LassoCV(cv=5, random_state=42).fit(train_X, train_y)
    return lasso

def evaluate_model(model: Union[LinearRegression, LassoCV], test_set: pd.DataFrame, colname: str) -> None:
    test_X = test_set.drop(colname, axis=1)
    test_y = test_set[colname]

    predictions = model.predict(test_X)
    mse = mean_squared_error(test_y, predictions)
    rmse = np.sqrt(mse)

    print("RMSE: ", rmse)

def run_train(df: pd.DataFrame, colname: str, type: str = 'linear') -> Tuple[Union[LinearRegression, LassoCV], pd.DataFrame, pd.DataFrame]:
    train_set, test_set = create_train_test_sets(df)
    if type == 'lasso':
        model = train_lasso_model(train_set, colname)
    else:
        model = train_linear_model(train_set, colname)
    return model, train_set, test_set

