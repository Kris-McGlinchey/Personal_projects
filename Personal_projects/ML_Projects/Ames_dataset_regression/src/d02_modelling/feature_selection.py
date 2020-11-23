def take_top_n_features(categorical, numerical, n, separate = True):

    if separate:
        categorical_sorted = {k: v for k, v in sorted(categorical.items(), key=lambda item: item[1])}
        numerical_sorted = {k: v for k, v in sorted(numerical.items(), key=lambda item: item[1])}
        numerical_columns = [k for k in list(numerical_sorted)[:n]]
        categorical_columns = [k for k in list(categorical_sorted)[:n]]
        total_columns = numerical_columns + categorical_columns
    else:
        total = {**categorical, **numerical}
        total_sorted = {k: v for k, v in sorted(total.items(), key=lambda item: item[1])}
        total_columns = [k for k in list(total_sorted)[:n]]
    return total_columns