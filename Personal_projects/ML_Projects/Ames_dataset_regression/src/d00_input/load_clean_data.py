import pandas as pd

COLUMNS_TO_DROP = ['PoolQC',
                  'MiscFeature',
                  'Alley']

def load_csv(filename):    
    data = pd.read_csv(filename)
    return data

def drop_columns(df):
    df.drop(COLUMNS_TO_DROP, inplace = True, axis = 1)
    return df