# Bremen Big Data Challenge 2019

This project proposes learning approach for the given classification problem presented in the Bremen Big Data Challenge competition 2019. The dataset comprises of data obtained from various sensors which are placed on the leg of human subjects which are sampled at 1000Hz. The aim of the learning model is to accurately classify the 22 different human activity labels. Data is presented in such a way that each activity of a subject is a CSV file that is made up of 19 features and many rows. The first step in our model building process is Data Pre-processing. Feature extraction methods are used to extract useful information from the raw data. Then various Feature selection approaches are employed in order to select best features for the model. After this, data exploratory analysis is performed for better interpretation of the data. With the help of appropriate visualization techniques exploratory data analysis is completed. Next phase is to do Data Analysis on the pre-processed data based on the insights gained in the data exploration phase. Using Skit-Learn’s GridSearchCV, we are able to select the best hyperparameters for three chosen models. Based on the cross-validation accuracy and the nature of problem, RandomForest model is selected as the model for disposition. A RndomForest model with obtained hyperparameters is trained with training data and the results are assessed carefully and the details are described in the report.

## Requirements

- [Scikit-learn](http://scikit-learn.org/stable/)
- [Python 3](https://www.python.org/)

## Dataset
dataset is available at: https://bbdc.csl.uni-bremen.de/
