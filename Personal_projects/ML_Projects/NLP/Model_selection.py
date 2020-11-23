import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import sys

if __name__ == '__main__':
    
    data = pd.read_csv(data_file)
    feature_columns = load('../param_store/selected_columns.joblib')
    
    X = data.drop('label', axis = 1)
    y = data['label']

    X = X['feature_columns']
    scale = StandardScaler()
    X = scale.fit_transform(X)
    transformations = np.vstack([sc.mean_, sc.scale_])
    dump(transformations, './param_store/scaling_factors.joblib')
    
    param_grid = {'max_depth': [30, 40, 50],
                  'n_estimators': [100, 200, 300],
                  'min_samples_split':  [2, 5, 10],
                  'min_samples_leaf' : [1, 2, 5, 10]}

    # Create a based model
    rf = RandomForestClassifier()
    # Instantiate the grid search model
    grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_grid, 
                                  cv = 5,
                                  return_train_score = True,
                                  scoring = 'f1')
    _ = grid_search_rf.fit(X, y.values)
    print('Best hyperparameters:')
    print(grid_search_rf.best_params_)
    rf_final = RandomForestClassifier(**grid_search_rf.best_params_)
    rf_final.fit(X_train, y_train.values)
    
    dump(rf_final, './param_store/rf_model.joblib')