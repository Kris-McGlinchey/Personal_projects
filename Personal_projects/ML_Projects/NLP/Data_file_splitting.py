import pandas as pd
from sklearn.model_selection import train_test_split

import sys

if __name__ == '__main__':
    data_file = sys.argv[1]
    
    data = pd.read_csv(data_file)
    X = data.drop('label', axis = 1)
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
    
    data_train = pd.concat([X_train, y_train], axis=1)
    data_test = pd.concat([X_test, y_test], axis=1)
    data_test.to_csv('./data/test_raw.csv', index = False)
    data_train.to_csv('./data/train_raw.csv', index = False)