import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from typing import List, Union

__all__ = []
__all__.extend([
    'plot_numeric_features_vs_log_price',
    'plot_categorical_features_vs_log_price',
    'plot_correlation_heatmap',
    'plot_geospatial_data',
    'visualize',
    'plot_learning_curves'
])

def plot_numeric_features_vs_log_price(df: pd.DataFrame, colname: str, num_columns: List[str]) -> None:
    for col in num_columns:
        plt.figure(figsize=(8, 5))
        plt.scatter(df[col], df[colname], alpha=0.3)
        plt.title(f'{col} vs {colname}')
        plt.xlabel(col)
        plt.ylabel(colname)
        plt.show()

def plot_categorical_features_vs_log_price(df: pd.DataFrame, colname: str, cat_columns: List[str]) -> None:
    for col in cat_columns:
        plt.figure(figsize=(12, 6))
        df.boxplot(column=colname, by=col)
        plt.title(f'{colname} by {col}')
        plt.xlabel(col)
        plt.ylabel(colname)
        plt.xticks(rotation=45)
        plt.suptitle('')
        plt.show()

def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> None:
    fig, ax = plt.figure(figsize=(12, 10)), plt.axes()
    cax = ax.matshow(corr_matrix, cmap='coolwarm')
    fig.colorbar(cax)

    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

    for (i, j), val in np.ndenumerate(corr_matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')

    plt.title('Correlation Matrix Heatmap', pad=20)
    plt.show()

def plot_geospatial_data(df: pd.DataFrame, colname: str, lat_col: str = 'latitude', lon_col: str = 'longitude') -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(df[lon_col], df[lat_col], c=df[colname], cmap='viridis', alpha=0.5)
    plt.colorbar(label=colname)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Airbnb Listings Geospatial Distribution')
    plt.show()

def visualize(df: pd.DataFrame, colname: str, num_columns: List[str], cat_columns: List[str], return_corr: bool = True) -> Union[pd.DataFrame, None]:
    plot_numeric_features_vs_log_price(df, colname, num_columns)
    plot_categorical_features_vs_log_price(df, colname, cat_columns)
    plot_geospatial_data(df, colname)
    if return_corr:
        corr_matrix = examine_correlations(df, colname)
        plot_correlation_heatmap(corr_matrix)
        return corr_matrix
    else:
        return None

def plot_learning_curves(model: LinearRegression, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
    train_sizes, train_scores, validation_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error')

    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)

    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Learning curves', fontsize=18, y=1.03)
    plt.legend()
    plt.ylim(0,3)

