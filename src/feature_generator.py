import numpy as np  # For mathematical calculations
import pandas as pd  # For Pandas DataFrame
from scipy.stats import kurtosis, skew  # To calculate skewness, kurtosis


def extract_features(df, feature, column_name):
    stats_df = pd.DataFrame()
    switcher_df = df
    stats_df[column_name] = time_stats(switcher_df, feature)
    return(stats_df.transpose())

def time_stats(switcher_df, switcher_feature):
    switcher={
        'mean': switcher_df.mean(),
        'median': switcher_df.median(),
        'min': switcher_df.min(),
        'max': switcher_df.max(),
        'std': switcher_df.std(),
        'variance': switcher_df.var(),
        'mad': switcher_df.mad(),
        'rms': np.sqrt(np.sum(np.power((switcher_df),2))/len(switcher_df)),
        'zcr': np.diff(np.signbit(switcher_df)).sum(),
        'iqr': switcher_df.quantile(0.75) - switcher_df.quantile(0.25),
        'pe': switcher_df.quantile(0.75),
        'kurtosis': kurtosis(switcher_df),
        'skew': skew(switcher_df)
     }
    return switcher.get(switcher_feature,"Invalid feature")
