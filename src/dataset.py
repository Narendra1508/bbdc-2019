import os
import warnings  # To ignore any warnings warnings.filterwarnings("ignore")
from glob import glob  # glob uses the wildcard pattern to create an iterable object file names # containing all matching file names in the current directory.

import numpy as np  # For mathematical calculations
import pandas as pd  # For Pandas DataFrame
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For data visualization

from . import extract_features


def main(train_label_df, feature):
    # files: to hold list of all datafiles to be read 
    folders=[]
    for sub in train_label_df.Datafile:
        folders.append(sub.split('/')[0])

    for folder in set(folders):
        # read all the training datasets one by one for each subject
        train_data_df = (fetch_train_data(folder , feature))

        train_df = pd.DataFrame(train_data_df)
        
        train_df['activity'] = train_label_df.loc[train_label_df.Subject == folder]['Label'].to_list()
        
        # write the file with extracted features 
        train_df.to_csv("./../dataset/processed_train/"+ folder + ".csv")

# Here is the function to fecth the summarised training data of each subject
def fetch_train_data(folder, feature):
    # list to hold each processed dataframe
    dataframes = []
    
    # filenames: holds all the activity files given subject
    file_names = glob("./../dataset/train/" + folder +"/*.csv")
    
    #read each activity file of the subject
    for file_name in file_names:
        df = pd.read_csv(file_name, header=None)
    
        #feature extraction of each activity file
        processed_df = extract_features(df, feature, file_name.split('\\')[1])
        
        #append the processed dataframe to dataframes list 
        dataframes.append(processed_df)
    
    return dataframes
    
    
if __name__ == '__main__':
    # read labled training data
    train_label_df = pd.read_csv("./../dataset/train.csv")
    
    # Select required feature from the below set
    # {'mean','median','min','max','std','variance','mad','rms','zcr','iqr','pe','kurtosis','skew'}
    feature = 'mean'
    main(train_label_df, feature)
