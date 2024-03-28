import numpy as np
from dataExploration import import_data, clean_string_column, explore_data, prep_data, visualize, remove_highly_correlated_features, run_train, feature_importance_from_model, evaluate_model, add_per_accommodates

def main():
    colname = 'log_price'
    df = import_data('Airbnb_data.csv')
    # df = df.head(20) # UNCOMMENT THIS LINE TO TEST ON A SUBSET OF THE DATA FOR SPEED

    # SIMPLE STRING CLEAN UP AND EXPLORATION
    df = df.apply(clean_string_column)
    explore_data(df)

    # CLEAN THE DATA
    cols_to_drop = ['id', 'amenities', 'description', 'first_review', 'host_has_profile_pic', 'host_response_rate', 'host_since', 'last_review', 'name', 'city', 'neighbourhood', 'property_type', 'room_type', 'thumbnail_url', 'zipcode'] #, 'bed_type', 'host_identity_verified', 'instant_bookable', 'cancellation_policy'
    df = prep_data(df, colname, cols_to_drop)

    # ADD SOME EXTRA FEATURES BEFORE PRECEDING
    cols = ['beds', 'bedrooms']
    for c in cols:
        df = add_per_accommodates(df, c)

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
    main()
