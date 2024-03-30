from dataclasses import asdict
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from simple_parsing import field, ArgumentParser

import numpy as np
from dataExploration.utils import import_data, set_seed, filter_data
from dataExploration.eda import explore_data
from dataExploration.processing import clean_string_column, prep_data
from dataExploration.visualization import visualize
from dataExploration.feature_engineering import remove_highly_correlated_features, feature_importance_from_model, add_per_col
from dataExploration.training import run_train, evaluate_model
from dataExploration.params import HyperParams

@dataclass
class Options:
    """ options """
    # .yml file containing HyperParams
    config_file: str = field(alias='-c', required=True)

    # where to load your data from
    input_file: str = field(alias='-f', required=True)

    # where to save training results
    output_file: str = field(alias='-o', required=True)

    # random seed
    seed: int = field(alias='-s', default=None, required=False)

def main(config_file: str, input_file:str, output_file: str, seed: int = None):
    if seed is not None:
        set_seed(seed)

    # import the data
    df = import_data(Path(os.getcwd() + input_file))

    # File locations and parameters
    hp = HyperParams.load(Path(os.getcwd() + config_file))
    colname = hp.colname
    cols_to_drop = hp.cols_to_drop
    per_colname = hp.per_colname
    per_cols = hp.per_cols
    keep = hp.keep

    # SIMPLE STRING CLEAN UP AND EXPLORATION
    df = df.apply(clean_string_column)
    if keep != '':
        df = filter_data(df, colname, keep)
    explore_data(df)

    # CLEAN THE DATA
    df = prep_data(df, colname, cols_to_drop)

    # ADD SOME EXTRA FEATURES BEFORE PRECEDING
    for c in per_cols:
        df = add_per_col(df, c, per_colname)

    # GET THE TYPES OF COLUMNS OF EACH TYPE FOR VISUALIZATION
    num_columns = df.select_dtypes(include=[np.number]).columns.drop('log_price').tolist()
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # VISUALIZATIONS
    print('-'*90)
    print('Here are our correlation results:')
    df, corr_matrix = visualize(df, colname, num_columns, cat_columns, return_corr=True)

    # REMOVE HIGHLY CORRELATED COLUMNS FROM BEFORE TRAINING
    if type(corr_matrix) != None:
        df, dropped_features = remove_highly_correlated_features(df, corr_matrix, threshold=0.9)
        print(f"Dropped Features due to high correlation: {dropped_features}")

    # RUN TRAINING
    model_str = "forrest"
    if model_str == "lasso":
        model_type = "LassoCV"
    elif model_str == "forrest":
        model_type = "RandomForestRegressor"
    elif model_str == "tree":
        model_type = "DecisionTreeRegressor"
    else:
        model_type = "LinearRegression"

    model, train_set, test_set  = run_train(df, colname, type=model_str)

    # SEE HOW THE MODEL RELATES TO THE FEATURES
    feature_names = [col for col in train_set.columns if col != 'log_price']
    importance = feature_importance_from_model(model, feature_names)
    print('-'*90)
    print('Here are our importance results:')
    print(importance)

    # EVALUATE THE MODEL
    evaluate_model(model, train_set, test_set, colname)
    print('-'*90)
    print(f'Here we have used the {model_type} model to make a model which predicts the {colname} based on a number of features within our dataset. Specifically, we have feature engineered lattitude and longitude to be combined into a density heatmap weighted by {colname} and used that new feature within our model to improve the results obtained by the example data pipeline. Additionally, we have made it possible to run our script via CLI which could be useful in automated pipelines. To improve this, we could accept command line arguments for the filename, column name which should be predicted, and which model should be used, and instead of showing plots, save them as files so no popup is generated -- thereby pausing the script until a user closes the figure. Lastly, I added type safety to all of the function arguments and returns to make sure some potential bugs can be caught through linting.')

if __name__ == "__main__":
    parser = ArgumentParser(add_dest_to_option_strings=False,
                            add_option_string_dash_variants=True)
    parser.add_arguments(Options, "options")
    args = parser.parse_args()

    main(**asdict(args.options))
