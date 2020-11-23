import pandas as pd

from joblib import dump, load

from src.d01_processing.data_conversion import convert_numeric_to_categorical
from src.d02_modelling.feature_engineering import replace_quality_labels, replace_garage_finish, get_dummies


if __name__ == '__main__': 
    scaling_factors = load('./param_store/scaling_factors.joblib')
    rf_model = load('./param_store/RfRegressor.joblib')
    feature_columns = load('./param_store/feature_columns.joblib')
    engineered_columns = load('./param_store/engineered_columns.joblib')

    training_data = pd.read_csv('./data/test.csv')

    data = training_data[feature_columns]
    data = convert_numeric_to_categorical(data)
    data = replace_quality_labels(data)
    data = replace_garage_finish(data)
    data = get_dummies(data)

    for col in data.select_dtypes(include = 'float'):
        data[col] = data[col].fillna(0)
        
    data = data[engineered_columns]
    data = (data - scaling_factors[0]) / scaling_factors[1]
    
    results = rf_model.predict(data)
    print(results)