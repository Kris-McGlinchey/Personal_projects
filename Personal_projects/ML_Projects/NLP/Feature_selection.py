import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

from joblib import dump, load
import sys

def mann_whitney_significance_test(df):
    pval_dict = {}
    for col in df.columns[:-1]:
        data = df.groupby('label')[col].apply(np.array).reset_index()
        pval = stats.mannwhitneyu(*data.iloc[:, 1].values)[1]
        pval_dict[col] = pval
    return pval_dict



if __name__ == '__main__':
    data_file = sys.argv[1]
    data = pd.read_csv(data_file)
    
    results = mann_whitney_significance_test(data)
    results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}

    # Standard protocol says to reject the null hypothesis if p < 0.05
    results_significant = {k: v for k, v in sorted(results.items(), key=lambda item: item[1]) if v < 0.05}
    significant_columns = list(results_significant.keys()) + ['label']
    data = data[significant_columns]
    
    # Create correlation matrix
    corr_matrix = data.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    data = data.drop(to_drop, axis = 1)
    
    X = data.drop('label', axis = 1)
    y = data['label']
    
    scale = StandardScaler()
    X = scale.fit_transform(X)
    X = pd.DataFrame(X_train, columns = labels)
    
    lr = LogisticRegression()
    rfecv = RFECV(estimator=lr, step=5, cv=5, scoring='f1')
    rfecv.fit(X, y)
    
    df_feature_importance_lr = pd.DataFrame(list(zip(labels, rfecv.support_)), columns = ['Feature', 'Importance'])
    important_columns = df_feature_importance_lr[df_feature_importance_lr['Importance'] == True]['Feature'].values
    dump(important_columns, './param_store/selected_columns.joblib')