import numpy as np
import scipy.stats as stats

def score_categorical_feature(X, y, columns, how):
    stat_test_dict = {}
    X['SalePrice'] = y
    for col in columns:
        data = X.groupby(col)['SalePrice'].apply(np.array).reset_index()
        if how == 'kruskal':
            stat_test_dict[col] = stats.kruskal(*data.iloc[:, 1].values)[1]
        elif how == 'anova':
            stat_test_dict[col] = stats.anova(*data.iloc[:, 1].values)[1]
    return stat_test_dict

def score_numerical_feature(X, y, columns, how):
    stat_test_dict = {}
    X['SalePrice'] = y
    for col in columns:
        if how == 'spearman':
            stat_test_dict[col] = stats.spearmanr(X[col].values, X['SalePrice'].values)[1]
        elif how == 'pearson':
            stat_test_dict[col] = stats.pearsonr(X[col].values, X['SalePrice'].values)[1]
    return stat_test_dict