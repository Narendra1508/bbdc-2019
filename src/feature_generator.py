import os
import errno
import warnings  # To ignore any warnings warnings.filterwarnings("ignore")
from glob import glob  # glob uses the wildcard pattern to create an iterable object file names # containing all matching file names in the current directory.

import numpy as np  # For mathematical calculations
import pandas as pd  # For Pandas DataFrame
from scipy.stats import kurtosis, skew  # To calculate skewness, kurtosis

import pickle # To store the necessary files for efficient reuse

def main(train_label_df, feature, data_flag):
    #list all subject's data floder
    folders = os.listdir("dataset/" + data_flag)
    
    df = pd.DataFrame()

    for folder in set(folders):
        # read all the datasets one by one for each subject in the directory listing
        data_df = pd.DataFrame(fetch_train_data(folder , feature))
        
        # add the labels to training data
        if(data_flag == 'train'):
            data_df['activity'] = train_label_df.loc[train_label_df.Subject == folder]['Label'].to_list()
        
        # append each subject's dataframe to a common dataframe
        df = df.append(data_df, ignore_index=True)

    # random shuffling of all test/train data
    df = df.sample(frac=1).reset_index(drop=True)

    # write the file with extracted features
    out_file = "dataset/pickle/"+ data_flag +"/"+ feature +".pickle"
    if not os.path.exists(os.path.dirname(out_file)):
        try:
            os.makedirs(os.path.dirname(out_file))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    pickle_out = open(out_file,"wb")
    pickle.dump(df, pickle_out)

    print("Excuted Successfully")

# Here is the function to fecth the summarised training data of each subject
def fetch_train_data(folder, feature):
    # DataFrame to hold each processed dataset
    dataframe = pd.DataFrame()
    
    # filenames: holds all the activity files given subject
    file_names = glob("dataset/" + data_flag + "/" + folder +"/*.csv")
    
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
    
    # flag is to determine test and train data
    data_flag = 'train'
    
    main(train_label_df, feature, data_flag)
