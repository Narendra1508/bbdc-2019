from glob import glob  # glob uses the wildcard pattern to create an iterable object file names # containing all matching file names in the current directory.

import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    train_df = pd.DataFrame()

    files = glob("dataset/processed_train/*_std.csv")
    
    # Append all the processed files to one dataframe
    for file in files:
        train_df = train_df.append(pd.read_csv(file), ignore_index=True)

    # random shuffling of all training data
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    train_df.to_csv("dataset/std_final_train.csv", index=False)

    print("Merged files successfully")
