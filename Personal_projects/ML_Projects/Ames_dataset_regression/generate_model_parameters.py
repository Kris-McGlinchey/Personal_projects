import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump, load

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from src.d01_processing.data_conversion import convert_numeric_to_categorical
from src.d02_modelling.feature_engineering import replace_quality_labels, replace_garage_finish, get_dummies
from src.d02_modelling.model_selection import hyperparameter_tuning_rf

if __name__ == '__main__':

    feature_columns = load('./param_store/feature_columns.joblib')

    training_data = pd.read_csv('./data/train.csv')
    
    X = training_data[feature_columns]
    X = convert_numeric_to_categorical(X)
    X = replace_quality_labels(X)
    X = replace_garage_finish(X)
    X = get_dummies(X)

    for col in X.select_dtypes(include = 'float'):
        X[col] = X[col].fillna(0)
    
    y = training_data['SalePrice']
    dump(X.columns, './param_store/engineered_columns.joblib')
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    transformations = np.vstack([sc.mean_, sc.scale_])
    dump(transformations, './param_store/scaling_factors.joblib')
    
    param_grid = {
    'max_features': ['auto'],
    'oob_score': [True],
    'max_depth': [20],
    'n_estimators': [200]
    }

    best_parameters = hyperparameter_tuning_rf(X, y, param_grid)
    
    rf_final = RandomForestRegressor(**best_parameters)
    rf_final.fit(X_train, y_train)
    dump(rf_final, './param_store/RfRegressor.joblib')
    results_rf = rf_final.predict(X_test)