# -*- coding: utf-8 -*-
import os
import numpy as np

def reduced_size(all_data, reduce):

# Numpy array for SVM for the next three parts:
# input_data_svm (n x 4) dimensional np-array with columns as (temp, strain rate, true strain, true stress)
    input_data_svm = np.array([[0, 0, 0, 0]])

    for i in range(0, len(all_data)):
        temp = all_data[i].T
        temp = temp[~np.any(temp == 0, axis  = 1)]
        temp = temp[~np.isnan(temp).any(axis=1)]
        if reduce is 0:
            input_data_svm = np.concatenate((input_data_svm, temp)) # This returns the whole input data as single numPy array
        else:
        # Reducing the number of data points
            if(temp.shape[0] > 400):
                y = temp[np.random.choice(temp.shape[0], 400, replace=False), :]
                input_data_svm = np.concatenate((input_data_svm,y))
            else:
                input_data_svm = np.concatenate((input_data_svm,temp))
    input_data_svm = input_data_svm[1:,:]
    return input_data_svm