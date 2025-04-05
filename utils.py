import numpy as np
from scipy import stats

# Funcții helper pentru tratarea valorilor extreme
def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for column in columns:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df_clean[~((df_clean[column] < (Q1 - 1.5 * IQR)) | (df_clean[column] > (Q3 + 1.5 * IQR)))]
    return df_clean



def remove_outliers_zscore(df, columns, threshold=3):
    df_clean = df.copy()
    z_scores = np.abs(stats.zscore(df_clean[columns]))
    df_clean = df_clean[(z_scores < threshold).all(axis=1)]
    return df_clean


def apply_log_transform(df, columns):
    df_transformed = df.copy()
    for col in columns:
        if not (df[col] <= 0).any():
            df_transformed[col] = np.log1p(df_transformed[col])
    return df_transformed