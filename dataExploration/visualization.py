import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from typing import Tuple, List, Union
from dataExploration.feature_engineering import density_lookup
import seaborn as sns

__all__ = []
__all__.extend([
    'plot_numeric_features_vs_log_price',
    'plot_categorical_features_vs_log_price',
    'plot_correlation_heatmap',
    'plot_geospatial_heatmap',
    'histogram_boxplot',
    'perc_on_bar',
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

def plot_geospatial_heatmap(df: pd.DataFrame, colname: str, lat_col: str = 'latitude', lon_col: str = 'longitude', weighted: bool = True, gridsize: int = 10000, cmap: str = 'viridis') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # OLD SIMPLE SCATTER - DEPRECATED
    # plt.figure(figsize=(10, 6))
    # plt.scatter(df[lon_col], df[lat_col], c=df[colname], cmap='viridis', alpha=0.5)
    # plt.colorbar(label=colname)
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.title('Airbnb Listings Geospatial Distribution')
    # plt.show()

    # MAKE A GRID MANUALLY
    lat_min, lat_max = df[lat_col].min(), df[lat_col].max()
    lon_min, lon_max = df[lon_col].min(), df[lon_col].max()
    lat_bins = np.linspace(lat_min, lat_max, gridsize+1)
    lon_bins = np.linspace(lon_min, lon_max, gridsize+1)

    plt.figure(figsize=(10, 6))
    if weighted:
        # GET DENSITY WEIGHTED BY THE COLNAME
        density, _, _ = np.histogram2d(df[lat_col], df[lon_col], bins=[lat_bins, lon_bins], weights=df[colname])

        # PLOTTING
        plt.imshow(np.log1p(density.T), origin='lower', extent=[lon_min, lon_max, lat_min, lat_max], aspect='auto', cmap='viridis')
        plt.colorbar(label=f'{colname} weighted Log Density')
    else:
        # plt.hexbin(df[lon_col], df[lat_col], gridsize=gridsize, cmap=cmap, bins='log')
        # GET DENSITY WEIGHTED BY THE COLNAME
        density, _, _ = np.histogram2d(df[lat_col], df[lon_col], bins=[lat_bins, lon_bins])

        # PLOTTING
        plt.imshow(np.log1p(density.T), origin='lower', extent=[lon_min, lon_max, lat_min, lat_max], aspect='auto', cmap='viridis')
        plt.colorbar(label='Log Density')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Airbnb Listings Geospatial Density')
    plt.show()

    return density, lat_bins, lon_bins

def histogram_boxplot(feature, figsize=(10,8), bins = None):
    """ Boxplot and histogram combined
    feature: 1-d feature array
    figsize: size of fig (default (9,8))
    bins: number of bins (default None / auto)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(nrows = 2, # Number of rows of the subplot grid= 2
                                           sharex = True, # x-axis will be shared among all subplots
                                           gridspec_kw = {"height_ratios": (.25, .75)},
                                           figsize = figsize
                                           ) # creating the 2 subplots
    sns.boxplot(feature, ax=ax_box2, showmeans=True, color='violet') # boxplot will be created and a star will indicate the mean value of the column
    sns.distplot(feature, kde=F, ax=ax_hist2, bins=bins,color = 'orange') if bins else sns.distplot(feature, kde=False, ax=ax_hist2,color='tab:cyan') # For histogram
    ax_hist2.axvline(np.mean(feature), color='purple', linestyle='--') # Add mean to the histogram
    ax_hist2.axvline(np.median(feature), color='black', linestyle='-') # Add median to the histogram

def perc_on_bar(df: pd.DataFrame, z: str) -> None:
    '''
    plot
    feature: categorical feature
    the function won't work if a column is passed in hue parameter
    '''

    total = len(df[z]) # length of the column
    plt.figure(figsize=(15,5))
    ax = sns.countplot(df[z],palette='Paired')
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total) # percentage of each class of the category
        x = p.get_x() + p.get_width() / 2 - 0.05 # width of the plot
        y = p.get_y() + p.get_height()           # hieght of the plot

        ax.annotate(percentage, (x, y), size = 12) # annotate the percantage
    plt.show() # show the plot

def visualize(df: pd.DataFrame, colname: str, num_columns: List[str], cat_columns: List[str], return_corr: bool = True) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    df.hist(figsize=(20,15))
    histogram_boxplot(data[colname])
    if len(num_columns) > 0:
        plot_numeric_features_vs_log_price(df, colname, num_columns)
    if len(cat_columns) > 0:
        plot_categorical_features_vs_log_price(df, colname, cat_columns)
    density, lat_bins, lon_bins = plot_geospatial_heatmap(df, colname, weighted=True)

    # LET'S SAVE THE DENSITY AS A NEW FEATURE TO INCLUDE IN THE CORRELATION
    df['density'] = df.apply(lambda row: density_lookup(density, lat_bins, lon_bins, row['latitude'], row['longitude']), axis=1)
    df = df.drop(['latitude', 'longitude'], axis=1)
    if return_corr:
        corr_matrix = examine_correlations(df, colname)
        plot_correlation_heatmap(corr_matrix)
        return df, corr_matrix
    else:
        return df

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

