from random import random
import sys
import logging
import mlflow.sklearn
import mlflow
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from pydrive.drive import GoogleAuth

import numpy as np
import pandas as pd
import dvc.api
import sys
import warnings
import os


# >> #### Import modules
# sys.path.append(os.path.abspath(os.path.join('../scripts')))
# from abtest-mlops2
# sys.path.insert(
#     0, '/home/jedi/Documents/Tenacademy/Week2/abtest_mlops2/scripts')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from preprocess import Preprocess
from ml import Ml


ml = Ml()
preprocess = Preprocess()

# path = 'data/AdSmartABdata.csv'
# repo = 'https://github.com/abtesting10academy/abtest-mlops'
# version = 'ea64953afc441740caf168dbfaf6ce3ad1bd1d16'

# data_url = dvc.api.get_url(
#     path=path,
#     repo=repo,
#     rev=version
# )


# data = pd.read_csv(data_url, sep=',')
# sys.path.insert(1, '/home/jedi/Documents/Tenacademy/Week2/abtest_mlops2/data/')
# import AdSmartABdata.csv
data = pd.read_csv('data/AdSmartABdata.csv', sep=',')


mlflow.set_experiment('SmartAD data analysis')

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    mlflow.log_param('data_url', data)
    mlflow.log_param('input_rows', data.shape[0])
    mlflow.log_param('input_cols', data.shape[1])
    mlflow.log_param('model_type','Random Forest')
    mlflow.log_param('model_parameters', 'n_estimators=100, max_depth=10, max_features=0.5')
    # change the date column to datetime
    data = preprocess.convert_to_datetime(data, 'date')
    numerical_column = preprocess.get_numerical_columns(data)
    categorical_column = preprocess.get_categorical_columns(data)

    # drop auction_id from categorical_column
    categorical_column.remove('auction_id')

    # Get column names have less than 10 more than 2 unique values
    to_one_hot_encoding = [col for col in categorical_column if data[col].nunique(
    ) <= 10 and data[col].nunique() > 2]

    # Get Categorical Column names thoose are not in "to_one_hot_encoding"
    to_label_encoding = [
        col for col in categorical_column if not col in to_one_hot_encoding]

    # Label encoding
    label_encoded_columns = preprocess.label_encode(data, to_label_encoding)

    # Select relevant rows

    # Copy our DataFrame to X variable
    X = data.copy()

    # Droping Categorical Columns,
    # "inplace" means replace our data with new one
    # Don't forget to "axis=1"
    X.drop(categorical_column, axis=1, inplace=True)

    # Merge DataFrames
    X = pd.concat([X, label_encoded_columns], axis=1)

    # Select only rows with responses
    X = X.query('yes == 1 | no == 1')

    # Drop auction_id column
    X.drop(["auction_id"], axis=1, inplace=True)

    # Split data

    X['target'] = [1] * X.shape[0]
    X.loc[X['no'] == 1, 'target'] = 0
    y = X['target']
    X.drop(["target"], axis=1, inplace=True)
    X.drop(['yes', 'no'], axis=1, inplace=True)

    # Get the day of the week from the date column as a new column
    X['day'] = X['date'].dt.dayofweek
    X.drop(["date"], axis=1, inplace=True)

    # IMport random forest model
    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=0)

    random_forest_result = ml.cross_validation(random_forest_model, X, y, 5)

    # Write scores to file
    with open("train/random_metrics.txt", 'w') as outfile:
        outfile.write(
            f"Training data accuracy: {random_forest_result['Training Accuracy scores'][0]}")
        outfile.write(
            f"Validation data accuracy: {random_forest_result['Validation Accuracy scores'][0]}")


    # Plot accuacy results to cml

    # Plot Accuracy Result
    model_name = "Random"
    ml.plot_result(model_name, "Accuracy", "Accuracy scores in 5 Folds",
                random_forest_result["Training Accuracy scores"],
                random_forest_result["Validation Accuracy scores"],
                'train/random_forest_accuracy.png')

    # Precision Results

    # Plot Precision Result
    ml.plot_result(model_name, "Precision", "Precision scores in 5 Folds",
                random_forest_result["Training Precision scores"],
                random_forest_result["Validation Precision scores"],
                'train/random_forest_preicision.png')

    # Recall Results plot

    # Plot Recall Result
    ml.plot_result(model_name, "Recall", "Recall scores in 5 Folds",
                random_forest_result["Training Recall scores"],
                random_forest_result["Validation Recall scores"],
                'train/random_forest_recall.png')


    # f1 Score Results

    # Plot F1-Score Result
    ml.plot_result(model_name, "F1", "F1 Scores in 5 Folds",
                random_forest_result["Training F1 scores"],
                random_forest_result["Validation F1 scores"],
                'train/random_forest_f1_score.png')


# The model is overfitting as it is working well on the training data but not on the validation set.
# We will adjust the min_samples_split hyperparameter to fix this.

# Fine tunin the min_samples_split parameter
# random_forest_model_2 = DecisionTreeClassifier(criterion="entropy",
#                                                min_samples_split=4,
#                                                random_state=0)
# random_forest_result_2 = ml.cross_validation(random_forest_model_2, X, y, 5)

# # Plot Accuracy Result
# ml.plot_result(model_name, "Accuracy", "Accuracy scores in 5 Folds",
#                random_forest_result_2["Training Accuracy scores"],
#                random_forest_result_2["Validation Accuracy scores"])


# # Plot Precision Result
# ml.plot_result(model_name, "precision", "precision scores in 5 Folds",
#                random_forest_result_2["Training Precision scores"],
#                random_forest_result_2["Validation Precision scores"])
# # Plot Recall Result
# ml.plot_result(model_name, "Recall", "Recall scores in 5 Folds",
#                random_forest_result_2["Training Recall scores"],
#                random_forest_result_2["Validation Recall scores"])


# # Plot F1-Score Result
# ml.plot_result(model_name, "F1", "F1 Scores in 5 Folds",
#                random_forest_result_2["Training F1 scores"],
#                random_forest_result_2["Validation F1 scores"])
