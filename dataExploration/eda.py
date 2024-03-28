import pandas as pd

__all__ = []
__all__.extend([
    'explore_data'
])

def explore_data(df: pd.DataFrame) -> None:
    # DROP THE ID COLUMN WHEN EXPLORING THE DATA, THIS WON'T EFFECT THE DF OUTSIDE OF THIS SCOPE
    df = df.drop(['id'], axis=1)

    # PRINT SAMPLE
    print('-'*90)
    print('Here is a sample of the data:')
    print(df.head())

    # PRINT DATASET INFO
    print('-'*90)
    print('This dataset has', df.shape[0], 'rows/observations, and ', df.shape[1], 'columns')
    print('The dataset has columns of types:')
    print(df.dtypes.value_counts())

    # PRINT VARIABLE INFO
    print('-'*90)
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].astype('category')

    cat_col = df.select_dtypes(include=['category'])
    for col in cat_col:
        print('Unique Values of {} are \n'.format(col),df[col].unique())
        print('-'*90)

    # PRINT DESCRIPTIVES
    print('Here are the descriptives of the dataset:')
    print(df.describe(include='all').T)


