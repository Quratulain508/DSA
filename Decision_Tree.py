# -*- coding: utf-8 -*-
"""
Created on Sun JUNE  29 10:00:12 2025

@author: Qurat
"""

# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
# train = pd.read_csv(train_url)
# # Function to import the dataset
def importdata():
    train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
    balance_data = pd.read_csv(train_url,sep=',', header=None)

    # Displaying dataset information
    print("Dataset Length: ", len(balance_data))
    print("Dataset Shape: ", balance_data.shape)
    print("Dataset: ", balance_data.head())
    
    # return balance_data
importdata()

# Function to split the dataset into features and target variables
def splitdataset(balance_data):
    train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
    balance_data = pd.read_csv(train_url,sep=',', header=None)
    # Separating the target variable
    X = balance_data.values[:, 1:5]
    Y = balance_data.values[:, 0]

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test