COLUMNS_TO_CONVERT_DTYPE = ['MSSubClass', 
                            'OverallQual',
                            'OverallCond', 
                            'KitchenAbvGr', 
                            'BedroomAbvGr', 
                            'HalfBath', 
                            'FullBath', 
                            'BsmtFullBath', 
                            'Fireplaces', 
                            'TotRmsAbvGrd']

def convert_numeric_to_categorical(df):
    for col in COLUMNS_TO_CONVERT_DTYPE:
        try:
            df[col] = df[col].astype('object')
        except:
            pass
    return df

def create_bins(df, col, splits, categories):
    df['_'.join([column, 'binned'])] = pd.cut(df[col], bins = splits, labels = categories)
    return