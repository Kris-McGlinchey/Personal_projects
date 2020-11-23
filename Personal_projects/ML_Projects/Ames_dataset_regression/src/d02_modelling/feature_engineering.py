import pandas as pd

COLUMNS_TO_REPLACE_LABELS = ['ExterQual', 'BsmtQual', 'KitchenQual', 'FireplaceQu', 'HeatingQC']
COLUMNS_TO_OHE = ['Neighborhood', 'GarageType', 'Foundation', 'MSSubClass']

def replace_quality_labels(df):
    for col in COLUMNS_TO_REPLACE_LABELS:
        try:
            df[col] = df[col].fillna('None')
            df[col] = df[col].map({'None': 0, 'Po':1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
        except:
            pass
    return df

def replace_garage_finish(df):
    df['GarageFinish'] = df['GarageFinish'].fillna('None')
    df['GarageFinish'] = df['GarageFinish'].map({'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})
    return df

def get_dummies(df):
    for col in COLUMNS_TO_OHE:
        OHE_data = pd.get_dummies(df[col], drop_first = True, prefix = col)
        df = pd.concat([df, OHE_data], axis = 1)
        df.drop(col, inplace = True, axis = 1)
    return df