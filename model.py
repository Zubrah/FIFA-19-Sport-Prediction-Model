# import neccessary  global libraries
import numpy as np  # linear algebra
import pandas as pd  # data processesing
import matplotlib.pyplot as plt  # data plotting and visualizations
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

from sklearn import tree, preprocessing
# ensembles
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import sklearn.metrics as metrics
# scores
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, auc
# models
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, learning_curve, GridSearchCV, \
    validation_curve
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import pickle
import os
import warnings

# -------------------------------------------------------------------------------------------------------

# data.csv has 18207 rows in reality, but we are only loading/previewing the first 1000 rows for preprocessing
nRowsRead = 5000  # specify 'None' if want to read whole file

FIFA_data = pd.read_csv('data.csv', index_col=[
    0], delimiter=',', nrows=nRowsRead)

# name the dataset
FIFA_data.dataframeName = 'data.csv'
nRow, nCol = FIFA_data.shape


# --------------------------------------------------------------------------------------------------
# Data Pre-Processing and Preparations..... 
# Imputation
def impute_data(df):
    df.dropna(inplace=True)


# Conversion weight to int
def weight_to_int(df):
    df['Weight'] = df['Weight'].str[:-3]
    df['Weight'] = df['Weight'].apply(lambda x: int(x))
    return df


# Conversion height to int
def height_convert(df_height):
    try:
        feet = int(df_height[0])
        dlm = df_height[-2]
        if dlm == "'":
            height = round((feet * 12 + int(df_height[-1])) * 2.54, 0)
        elif dlm != "'":
            height = round((feet * 12 + int(df_height[-2:])) * 2.54, 0)
    except ValueError:
        height = 0
    return height


def height_to_int(df):
    df['Height'] = df['Height'].apply(height_convert)


# One Hot Encoding of a feature
def one_hot_encoding(df, column):
    encoder = preprocessing.LabelEncoder()
    df[column] = encoder.fit_transform(df[column].values)


# Drop columns that we are not interested in
def drop_columns(df):
    df.drop(df.loc[:, 'ID':'Name'], axis=1, inplace=True)
    df.drop(df.loc[:, 'Photo':'Special'], axis=1, inplace=True)
    df.drop(df.loc[:, 'International Reputation':'Real Face'], axis=1, inplace=True)
    df.drop(df.loc[:, 'Jersey Number':'Contract Valid Until'], axis=1, inplace=True)
    df.drop(df.loc[:, 'LS':'RB'], axis=1, inplace=True)
    df.drop(df.loc[:, 'GKDiving':'Release Clause'], axis=1, inplace=True)


# Transform positions to  different valid  categories
def transform_positions(df):
    for i in ['ST', 'CF', 'LF', 'LS', 'LW', 'RF', 'RS', 'RW']:
        df.loc[df.Position == i, 'Position'] = 'Striker'
    for i in ['CAM', 'LAM', 'RAM']:
        df.loc[df.Position == i, 'Position'] = ' Attacking Midfielder'

    for i in ['CDM', 'LDM', 'RDM']:
        df.loc[df.Position == i, 'Position'] = 'Defensive Midfielder'

    for i in ['LCM', 'CM', 'LM', 'RCM', 'RM']:
        df.loc[df.Position == i, 'Position'] = 'Zone Midfielder '

    for i in ['CB', 'LCB', 'RCB']:
        df.loc[df.Position == i, 'Position'] = 'Central Defender'

    for i in ['LB', 'LWB', 'RB', 'RWB']:
        df.loc[df.Position == i, 'Position'] = 'Wings Backs'

    for i in ['GK']:
        df.loc[df.Position == i, 'Position'] = 'GoalKeeper'


# Drop columns that we are not interested in
drop_columns(FIFA_data)
# Impute the data that is null
impute_data(FIFA_data)
# transform weight and height to integer values
weight_to_int(FIFA_data)
height_to_int(FIFA_data)
# apply the one hot encoding to the Preferred foot (L,R) => (0,1)
one_hot_encoding(FIFA_data, 'Preferred Foot')
# transform position to striker, midfielder, defender
transform_positions(FIFA_data)
# show the 10 first rows
FIFA_data.head(10)

FIFA_data2 = FIFA_data

# ------------------------------------------------------------------------------
# Divide and train the models.

# Drop the elements that has been created for Finishing, Strength and Freekick_Accurancy
drop_elements = ['Age', 'Preferred Foot', 'Weight',
                 'HeadingAccuracy', 'ShortPassing', 'Volleys',
                 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
                 'SprintSpeed', 'Reactions', 'Balance',
                 'Stamina', 'Strength', 'LongShots',
                 'Interceptions', 'Penalties', 'Composure',
                 'Marking', 'StandingTackle', 'SlidingTackle']

FIFA_data2 = FIFA_data2.drop(drop_elements, axis=1)

# Generate the unique values for the positions encoded as Defender:0, Midfielder:1, Striker:2
positions = FIFA_data2["Position"].unique()
encoder = preprocessing.LabelEncoder()
FIFA_data2['Position'] = encoder.fit_transform(FIFA_data2['Position'])

# The Y feature is the position
y = FIFA_data2["Position"]

# The other features are all but the position
FIFA_data2.drop(columns=["Position"], inplace=True)

# Split the data
X_train_dev, X_test, y_train_dev, y_test = train_test_split(FIFA_data2, y,
                                                            test_size=0.20,
                                                            random_state=42)


# -------------------------------
# function to calculate the score
def train_and_score(clf, X_train, y_train, X_test, y_test):
    clf = clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    # cf = confusion_matrix(y_test,preds)

    # print(plot_confusion_matrix(cf, class_names=positions))

    print(" Accuracy: ", accuracy_score(y_test, preds))
    print(" F1 score: ", metrics.f1_score(y_test, preds, average='weighted'))


LR = LogisticRegressionCV(cv=5, random_state=20, solver='lbfgs',
                          multi_class='multinomial', max_iter=5000)
train_and_score(LR, X_train_dev, y_train_dev, X_test, y_test)

# plot_learning_curve(LR, "Logistic Regression Curve", X_train_dev, y_train_dev)


# fit the model
LR.fit(FIFA_data2, y)

# Saving file in a pickle dump
pickle.dump(LR, open('model.pkl', 'wb'))

# Load model to compare results
model = pickle.load(open('model.pkl', 'rb'))

# print(LR.fit(FIFA_data2, y))

# predict using LR
LR.predict(FIFA_data2)
