import numpy as np
from dataExploration import import_data, clean_string_column, explore_data, prep_data, visualize, remove_highly_correlated_features, run_train, feature_importance_from_model, evaluate_model

def main():
    colname = 'log_price'
    df = import_data('Airbnb_data.csv')

    # SIMPLE STRING CLEAN UP AND EXPLORATION
    df = df.apply(clean_string_column)
    explore_data(df)

    # CLEAN THE DATA
    cols_to_drop = ['amenities', 'description', 'first_review', 'host_has_profile_pic', 'host_identity_verified', 'host_response_rate', 'host_since', 'last_review', 'name']
    df = prep_data(df, colname, cols_to_drop)

    # GET THE TYPES OF COLUMNS OF EACH TYPE FOR VISUALIZATION
    num_columns = df.select_dtypes(include=[np.number]).columns.drop('log_price').tolist()
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # VISUALIZATIONS
    corr_matrix = visualize(df, colname, num_columns, cat_columns, return_corr=True)

    # REMOVE HIGHLY CORRELATED COLUMNS FROM BEFORE TRAINING
    if corr_matrix != None:
        df, dropped_features = remove_highly_correlated_features(df, corr_matrix, threshold=0.95)
        print(f"Dropped Features due to high correlation: {dropped_features}")

    # RUN TRAINING
    model, train_set, test_set  = run_train(df, colname)

    # SEE HOW THE MODEL RELATES TO THE FEATURES
    feature_names = [col for col in train_set.columns if col != 'log_price']
    importance = feature_importance_from_model(model, feature_names)
    print(importance)

    # EVALUATE THE MODEL
    evaluate_model(model, test_set, colname)


if __name__ == "__main__":
    main()
