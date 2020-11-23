import numpy as np

from joblib import dump

from src.d00_input.load_clean_data import load_csv, drop_columns
from src.d01_processing.feature_ranking import score_categorical_feature, score_numerical_feature
from src.d01_processing.data_conversion import convert_numeric_to_categorical, create_bins
from src.d02_modelling.feature_selection import take_top_n_features


if __name__ == '__main__':

    raw_data = load_csv('./data/train.csv')
    raw_data = drop_columns(raw_data)
    X = raw_data.drop(['SalePrice'], axis = 1)
    y = raw_data['SalePrice']
    
    X = convert_numeric_to_categorical(X)
    
    categorical_columns = X.select_dtypes(include = 'object').columns
    categorical_pvalues = score_categorical_feature(X, y, categorical_columns, how = 'kruskal')
    numerical_columns = X.select_dtypes(exclude = 'object').columns
    numerical_pvalues = score_numerical_feature(X, y, numerical_columns, how = 'spearman')
    features_to_extract = take_top_n_features(categorical_pvalues, numerical_pvalues, 10, separate = True)
    features_to_extract.remove('SalePrice')
    features_to_extract.remove('Id')
    dump(features_to_extract, './param_store/feature_columns.joblib')