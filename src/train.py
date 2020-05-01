import joblib
import numpy as np
import pandas as pd
from sklearn import metrics, model_selection, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pickle

import xgboost as xgb

if __name__ == "__main__":
    feature = 'mean'
    data_flag = 'train'
    out_file = "./../dataset/pickle/"+ data_flag +"/"+ feature +".pickle"
    pickle_in = open(out_file, "rb")
    df = pickle.load(pickle_in)

    

    y_train = df.activity.values
    
    x_train = df.drop(["activity"], axis=1)
    
    le = preprocessing.LabelEncoder()
    
    y_train = le.fit_transform(y_train)

    pipeline_lr = Pipeline([('scaler1',StandardScaler()), ('clf',LogisticRegression(random_state=42))])
    
    pipeline_randomforest = Pipeline([('scaler2',StandardScaler()), ('clf',RandomForestClassifier())])

    pipeline_xgboost = Pipeline([('scaler2',StandardScaler()), ('clf', xgb.XGBClassifier())])
    
    model_pipeline = [pipeline_lr, pipeline_randomforest, pipeline_xgboost]
    
    # Dictionary of pipelines and classifier types for ease of reference
    pipe_dict = {0: 'Logistic Regression', 1: 'RandomForest', 2: 'XGboost'}
    
    parameters = [{
                     'clf__penalty': ['l2'],
                     'clf__C': np.logspace(0, 4, 10)
                    },
                    {
                     'clf__n_estimators': [10, 30],
                     'clf__max_features': [0.25, 1.0]
                    },
                    {'clf__nthread': [4], #when use hyperthread, xgboost may become slower
                    'clf__objective': ['binary:logistic'],
                    'clf__learning_rate': [0.05], #so called `eta` value
                    'clf__max_depth': [6],
                    'clf__min_child_weight': [11],
                    'clf__silent': [1],
                    'clf__subsample': [0.8],
                    'clf__colsample_bytree': [0.7],
                    'clf__n_estimators': [5], #number of trees, change it to 1000 for better results
                    'clf__seed': [1337]}
                   # {'estimator':[Any_other_estimator_you_want],
                   #  'estimator__valid_param_of_your_estimator':[valid_values]}
                  ]
    
    # fit the pipeline with the training data
    for model, pipe in zip(model_pipeline, pipe_dict):
        grid_search = GridSearchCV(estimator=model, param_grid=parameters[pipe], cv = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=42) )
        grid_search.fit(x_train, y_train)

        joblib.dump(grid_search, f"models/{pipe_dict[pipe]}.pkl")

        print("{}: model best parameters are --> {}".format(pipe_dict[pipe], grid_search.best_params_))

        print("{}: cv accuracy is  {}".format(pipe_dict[pipe], grid_search.best_score_))
