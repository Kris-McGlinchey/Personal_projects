from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def hyperparameter_tuning_rf(X, y, param):

    # Create a based model
    rf = RandomForestRegressor()
    # Instantiate the grid search model
    grid_search_rf = GridSearchCV(estimator = rf, param_grid = param, 
                              cv = 5)
    _ = grid_search_rf.fit(X, y)
    best_params = grid_search_rf.best_params_
    return best_params