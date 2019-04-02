###Explore More About the business problem from the given kaggle link

#     https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection




import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from skelarn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import AdaboostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

dataset = pd.read_csv("train_sample.csv")

# Basic exploratory data analysis 

# Number of unique values in each column
def unique_vals(x):
    return len(dataset[x].unique())

number_unique_vals = {x: unique_vals(x) for x in dataset.columns}
print(number_unique_vals)

##Feature Engineering

#Creating DateTime Based Features

def dateTimeFeatures(dataframe):
    # Derive new features using the click_time column
    dataframe['datetime'] = pd.to_datetime(dataframe['click_time'])
    dataframe['day_of_week'] = dataframe['datetime'].dt.dayofweek
    dataframe["day_of_year"] = dataframe["datetime"].dt.dayofyear
    dataframe["month"] = dataframe["datetime"].dt.month
    dataframe["hour"] = dataframe["datetime"].dt.hour
    return dataframe
	
#Creating IP Based Features

def grouped_features(df):
    # ip_count
    ip_count = df.groupby('ip').size().reset_index(name='ip_count').astype('uint16')
    ip_day_hour = df.groupby(['ip', 'day_of_week', 'hour']).size().reset_index(name='ip_day_hour').astype('uint16')
    ip_hour_channel = df[['ip', 'hour', 'channel']].groupby(['ip', 'hour', 'channel']).size().reset_index(name='ip_hour_channel').astype('uint16')
    ip_hour_os = df.groupby(['ip', 'hour', 'os']).channel.count().reset_index(name='ip_hour_os').astype('uint16')
    ip_hour_app = df.groupby(['ip', 'hour', 'app']).channel.count().reset_index(name='ip_hour_app').astype('uint16')
    ip_hour_device = df.groupby(['ip', 'hour', 'device']).channel.count().reset_index(name='ip_hour_device').astype('uint16')
    
    # merge the new aggregated features with the df
    df = pd.merge(df, ip_count, on='ip', how='left')
    del ip_count
    df = pd.merge(df, ip_day_hour, on=['ip', 'day_of_week', 'hour'], how='left')
    del ip_day_hour
    df = pd.merge(df, ip_hour_channel, on=['ip', 'hour', 'channel'], how='left')
    del ip_hour_channel
    df = pd.merge(df, ip_hour_os, on=['ip', 'hour', 'os'], how='left')
    del ip_hour_os
    df = pd.merge(df, ip_hour_app, on=['ip', 'hour', 'app'], how='left')
    del ip_hour_app
    df = pd.merge(df, ip_hour_device, on=['ip', 'hour', 'device'], how='left')
    del ip_hour_device
    
    return df
	
# create x and y train
X = train_sample.drop('is_attributed', axis=1)
y = train_sample[['is_attributed']]

# split data into train and test/validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
'''
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
'''
##AdaBoost Classifier with hyperparameter tuning

parameters = {"base_estimator__max_depth" : [2, 5],
              "n_estimators": [200, 400, 600]
             }
# base estimator
tree = DecisionTreeClassifier()

# adaboost with the tree as base estimator
# learning rate is arbitrarily set to 0.6, we'll discuss learning_rate below
clf = AdaBoostClassifier(
    base_estimator=tree,
    learning_rate=0.6,
    algorithm="SAMME")			 

grid_search_clf = GridSearchCV(clf, 
                               cv = 10,
                               param_grid=parameters, 
                               scoring = 'roc_auc', 
                               return_train_score= True,                         
                               verbose = 1)
							   
grid_search_clf.fit(X_train, y_train)
cv_results = pd.DataFrame(grid_search_clf.cv_results_)
print(cv_results)
# model performance on test data with chosen hyperparameters

# base estimator
tree = DecisionTreeClassifier(max_depth=2)

# adaboost with the tree as base estimator
clf = AdaBoostClassifier(
    base_estimator=tree,
    learning_rate=0.6,
    n_estimators=200,
    algorithm="SAMME")

clf.fit(X_train, y_train)

predictions = clf.predict_proba(X_test)

roc_score = roc_auc_score(y_test, predictions[:, 1])
print(roc_score)
















































