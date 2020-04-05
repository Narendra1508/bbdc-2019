import os
import warnings  # To ignore any warnings warnings.filterwarnings("ignore")
from glob import glob  # glob uses the wildcard pattern to create an iterable object file names # containing all matching file names in the current directory.

import numpy as np  # For mathematical calculations
import pandas as pd  # For Pandas DataFrame
from scipy.stats import kurtosis, skew  # To calculate skewness, kurtosis


def main(train_label_df, feature):
    # files: to hold list of all datafiles to be read 
    folders=[]
    for sub in train_label_df.Datafile:
        folders.append(sub.split('/')[0])

    for folder in set(folders):
        # read all the training datasets one by one for each subject
        train_data_df = pd.DataFrame(fetch_train_data(folder , feature))
        
        # add the labels to training data
        train_data_df['activity'] = train_label_df.loc[train_label_df.Subject == folder]['Label'].to_list()
        
        # write the file with extracted features 
        train_data_df.to_csv("dataset/processed_train/"+ folder + ".csv", index=False)

    print("Excuted Successfully")

# Here is the function to fecth the summarised training data of each subject
def fetch_train_data(folder, feature):
    # DataFrame to hold each processed dataset
    dataframe = pd.DataFrame()
    
    # filenames: holds all the activity files given subject
    file_names = glob("dataset/train/" + folder +"/*.csv")
    
    #read each activity file of the subject
    for file_name in file_names:
        df = pd.read_csv(file_name, header=None)
            
        #append the processed dataset to dataframe 
        dataframe = dataframe.append(extract_features(df, feature), ignore_index=True)
    
    return dataframe
    
def extract_features(df, feature):
    stats_df = pd.DataFrame()
    switcher_df = df
    stats_df = time_stats(switcher_df, feature)  
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

    
if __name__ == '__main__':
    # read labled training data
    train_label_df = pd.read_csv("dataset/train.csv")
    
    # Select required feature from the below set
    # {'mean','median','min','max','std','variance','mad','rms','zcr','iqr','pe','kurtosis','skew'}
    feature = 'mean'
    main(train_label_df, feature)
