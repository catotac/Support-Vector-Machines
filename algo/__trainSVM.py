# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Training the model using Temperature as the target
def trainSVM_temp(inputdata):
    f = open("tempSVMlog.txt", "a")
    temp_labels = inputdata[:,0]
    X_data = inputdata[:,1:]
    X_scaled = preprocessing.scale(X_data)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, temp_labels, test_size=0.1, random_state=0)
    print("### Training SVM Parameters ####\n")
    f.write("### Training SVM Parameters ####\n")
    parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
                'C': [1, 10, 100, 1000]},
              {'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'degree':[3]}]
    clf = GridSearchCV(SVR(), parameters, cv=5)
    clf.fit(X_train, y_train)
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        f.write("%0.3f (+/-%0.03f) for %r \n" % (mean, std * 2, params))
    f.write("\n The best parameters are %r \n" %(clf.best_params_))
    f.write("\n The score for the Test set are %0.3f \n" % (clf.score(X_test, y_test)))

    f.close()

# Training the model using Strain rate as the target
def trainSVM_srRate(inputdata):
    f = open("srRatelog.txt", "a")
    strain_labels = inputdata[:,1]
    X_data = inputdata[:,[0,2,3]]
    X_scaled = preprocessing.scale(X_data)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, temp_labels, test_size=0.1, random_state=0)
    f.write("### Training SVM Parameters ####\n")
    parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
                'C': [1, 10, 100, 1000]},
              {'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'degree':[3]}]
    clf = GridSearchCV(SVR(), parameters, cv=5)
    clf.fit(X_train, y_train)
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        f.write("%0.3f (+/-%0.03f) for %r \n" % (mean, std * 2, params))
    f.write("\n The best parameters are %r \n" % (clf.best_params_))
    f.write("\n The score for the Test set are %0.3f \n" % (clf.score(X_test, y_test)))
    f.close()