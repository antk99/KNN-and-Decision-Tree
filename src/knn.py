#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 14:00:26 2022

@author: lunadana
Sources : 
    - https://www.dataquest.io/blog/k-nearest-neighbors-in-python/

1- Setting the value for dependent and independent variables
2 - Splitting the dataset
3 - Fitting the KNN model
4 - Predict the test set

isssues to be resolved : 
    - Both categorical and continuous variables : to solve 
        - Solution 1 (nul) : Treat continuous as categorical, setting threshold
        - Solution 2 : Distance function specific to each variable type
        - Solution 3 (debrouille mais pas optimal) : Assuming we normalize between 0 and 1 
            then we could keep the euclidian for categorical since same order
    - Some continuous variables have different range and order of values, set all of them
    to have the same order by normalizing

"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace         #for debugging 

dataset = pd.read_csv("/Users/lunadana/Desktop/COMP551/MiniProject/hepatitis_clean.csv")
dataset_predictors = dataset.loc[:, dataset.columns != 'Class']
dataset_predictors_cat_only = dataset_predictors[]


# Distance definition
euclidean = lambda x1, x2: np.sqrt(np.sum((x1 - x2)**2, axis=-1))
manhattan = lambda x1, x2: np.sum(np.abs(x1 - x2), axis=-1)
hamilton = lambda x1, x2: np.sum(1 if x1!=x2 else 0, axis=-1)

# Data
# np.array([tuple(r) for r in dataset[['Age','Protime']].to_numpy()])
x, y = np.array([tuple(r) for r in dataset_predictors.to_numpy()]), np.array(dataset['Class'])                                   #slices the first two columns or features from the data

#print the feature shape and classes of dataset 
(N,D), C = x.shape, np.max(y)+1
print(f'instances (N) \t {N} \n features (D) \t {D} \n classes (C) \t {C}')

inds = np.random.permutation(N)                                                     #generates an indices array from 0 to N-1 and permutes it 

#split the dataset into train and test
x_train, y_train = x[inds[:10]], y[inds[:10]]
x_test, y_test = x[inds[10:]], y[inds[10:]]

#visualization of the data
#plt.scatter(x_train[:,0], x_train[:,1], c=y_train, marker='o', label='train')
#plt.scatter(x_test[:,0], x_test[:,1], c=y_test, marker='s', label='test')
#plt.legend()
#plt.ylabel('sepal length')
#plt.xlabel('sepal width')
#plt.show()


# KNN Class
class KNN:
    def __init__(self, K=1, dist_fn= euclidean):
        self.dist_fn = dist_fn
        self.K = K
        return
    def fit(self, x, y):
        ''' Store the training data using this method as it is a lazy learner'''
        self.x = x
        self.y = y
        self.C = np.max(y)+1 
        return self
    def predict(self, x_test):
        ''' Makes a prediction using the stored training data and the test data given as argument'''
        num_test = x_test.shape[0]
        #calculate distance between the training & test samples and returns an array of shape [num_test, num_train]
        distances = self.dist_fn(self.x[None,:,:], x_test[:,None,:])
        #ith-row of knns stores the indices of k closest training samples to the ith-test sample 
        knns = np.zeros((num_test, self.K), dtype=int)
        #ith-row of y_prob has the probability distribution over C classes
        y_prob = np.zeros((num_test, self.C))
        for i in range(num_test):
            knns[i,:] = np.argsort(distances[i])[:self.K]
            y_prob[i,:] = np.bincount(self.y[knns[i,:]], minlength=self.C) #counts the number of instances of each class in the K-closest training samples
        #y_prob /= np.sum(y_prob, axis=-1, keepdims=True)
        #simply divide by K to get a probability distribution
        y_prob /= self.K
        return y_prob, knns



model = KNN(K=4-1)
y_prob, knns = model.fit(x_train, y_train).predict(x_test)
print('knns shape:', knns.shape)
print('y_prob shape:', y_prob.shape)

#To get hard predictions by choosing the class with the maximum probability
y_pred = np.argmax(y_prob,axis=-1)
accuracy = np.sum(y_pred == y_test)/y_test.shape[0]

print(f'accuracy is {accuracy*100:.1f}.')

#boolean array to later slice the indexes of correct and incorrect predictions
correct = y_test == y_pred
incorrect = np.logical_not(correct)

#visualization of the points
plt.scatter(x_train[:,0], x_train[:,1], c=y_train, marker='o', alpha=.2, label='train')
plt.scatter(x_test[correct,0], x_test[correct,1], marker='.', c=y_pred[correct], label='correct')
plt.scatter(x_test[incorrect,0], x_test[incorrect,1], marker='x', c=y_test[incorrect], label='misclassified')

#connect each node to k-nearest neighbours in the training set
for i in range(x_test.shape[0]):
    for k in range(model.K):
        hor = x_test[i,0], x_train[knns[i,k],0]
        ver = x_test[i,1], x_train[knns[i,k],1]
        plt.plot(hor, ver, 'k-', alpha=.1)
    
plt.ylabel('sepal length')
plt.xlabel('sepal width')
plt.legend()
plt.show()











