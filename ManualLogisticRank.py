from __future__ import division
import numpy as np
import os
from collections import defaultdict
import time
from sklearn import linear_model, datasets
from sklearn.metrics import *
from sklearn.datasets import load_iris
from sklearn import preprocessing
from scipy.misc import logsumexp
#from tensorflow.examples.tutorials.mnist import input_data

# Functions to extrac the documents, query and rank information
def extractFeatures(split):
    features = []
    for i in xrange(2, 138):
        features.append(float(split[i].split(':')[1]))
    # Convert to tuples:
    return features

def extractQueryData(split):
    # Add tuples:
    queryFeatures = split[1].split(':')[1]
    return queryFeatures

def readDataset(path):
    dictio_quid= defaultdict(list)
    features_list=[]
    rank_list=[]
    print('Reading training data from file...')
    k=0
    with open(path, 'r') as file:
        for line in file:
            # Getting features:
            split = line.split()
            features=extractFeatures(split)
            features_list.append(features)

            # Getting the query:
            query = int(extractQueryData(split))
            #print "query: "+str(query)
            # Getting rank:
            rank = int(split[0])
            rank_list.append(rank)

            # Feeding dictionary:
            dictio_quid[query].append((features, rank))
            k+=1
            #if k==10:
            #   break
    print('Number of query ID %d' %(len(features_list)))
    return np.array(features_list),np.array(rank_list),dictio_quid

# Normalization:

def normalize_features(features):
    # Substracting the mean:
    mean_features=np.mean(features,axis=0)
    features=features-mean_features

    # Dividing by the std:
    std_features=np.std(features,axis=0)
    features=features/std_features
    print "features normalized"
    return features,mean_features,std_features

def get_class_weight(labels):
    class_weight={}
    total_num=len(labels)
    set_labels=set(labels)
    for i in set(labels):
        label_num = len(labels[labels==i])
        class_weight[i]=total_num/(len(set_labels)*label_num)
    return class_weight


# Scaling the function
def scaling(x):
    x=x-np.mean(x,0)
    x=x/np.std(x,0)
    return x

# Adding Offset:
def add_offset(x):
    n,d=x.shape
    return np.hstack((np.ones((n,1)),x))

# Initialization:
def init_weights(x,y):
    dim = x.shape[1]
    num_class=y.shape[1]
    return np.random.randn(dim, num_class)/dim

# Loss:

# correct solution:
def softmax(x,weights):
    a=x.dot(weights)

    #np.exp((a)-logsumexp(a))
    e_a = np.exp(a)
    sum_e_a = np.sum(e_a,1).reshape(x.shape[0],1)
    return e_a/sum_e_a# only difference

def crossEntropyLoss(weights,x,y):
    return -np.sum(y*np.log(softmax(x,weights)))

def softmax_derivative(x,weights):
    return softmax(x,weights)*(1-softmax(x,weights))

def optimize_weights(x,y,alpha=0.00001,num_it=5000):
    weights=init_weights(x,y)
    k=0
    for j in xrange(num_it):
        for i in xrange(weights.shape[1]):
            last_gradient = gradientLoss(y, x, weights, i)
            weights[:,i] -= -alpha*gradientLoss(y,x,weights,i)
        k+=1
        if k%100==0:
            print "loss after %d interations : %0.2f "%(k,crossEntropyLoss(weights,x,y))
        #print "weights: "+str(weights)
    return weights

def nesterov_optimization(x,y,alpha=0.00001,num_it=100):
    weights=init_weights(x,y)
    k=0
    for j in xrange(num_it):
        for i in xrange(weights.shape[1]):
            new_grad = 4
            weights[:,i] -= alpha*gradientLoss(y,x,weights,i)
        k+=1
        if k%100==0:
            print "loss after %d interations : %0.2f "%(k,crossEntropyLoss(weights,x,y))
        #print "weights: "+str(weights)
    return weights


def gradientLoss(t,x,weights,i):
    y=softmax(x,weights)
    dif=(t-y)
    dif_i=dif[:,i].reshape(dif.shape[0],1)
    soft_der=softmax_derivative(x, weights)
    soft_der=soft_der[:, i].reshape(soft_der.shape[0],1)
    grad=dif_i*soft_der*x
    return np.sum(grad,0)
# Making predictions and testing:

def predict(weights,x,y):
    return np.argmax(softmax(x,weights),1)

def get_errors(t,y):
    return np.sum(t==y)/t.shape[0]

if __name__=="__main__":

    time_start = time.clock()
    # Paths:
    training_path='./MSLR-WEB10K/Fold1/train.txt'
    validation_path='./MSLR-WEB10K/Fold1/vali.txt'

    #Read training data
    X_train,y_train, dictio_train= readDataset(training_path)
    X_train,_,_=normalize_features(X_train)

    x_sample=X_train[0:10000]
    x_sample=add_offset(x_sample)
    y_sample=y_train[0:10000]
    y_label=y_train[0:10000]

    lb = preprocessing.LabelBinarizer()
    lb.fit(y_sample)
    y_sample=lb.transform(y_sample)

    # Variables:
    n=x_sample.shape[0]
    dim=x_sample.shape[1]

    weights=optimize_weights(x_sample,y_sample)

    y_hat = predict(weights, x_sample, y_sample)

    # Test set:
    x_sample_test = X_train[1000:1500]
    x_sample_test = add_offset(x_sample_test)
    y_test_labels= y_train[1000:1500]
    y_test = lb.transform(y_test_labels)

    y_hat_test= predict(weights, x_sample_test, y_test)
    print get_errors(y_test_labels,y_hat_test)




    # # Using iris dataset:
    #
    # data = load_iris()
    # x=data.data
    # x=add_offset(x)
    # y=data.target
    #
    # # Encoding:
    # lb = preprocessing.LabelBinarizer()
    # lb.fit(y)
    # t=lb.transform(y)
    #
    # # Variables:
    # n=x.shape[0]
    # dim=x.shape[1]
    #
    # # Initialize weights:
    # weights=optimize_weights(x,t)
    #
    # # Predictions:
    # y_hat=predict(weights,x,t)
    #
    # error=get_errors(y,y_hat)
    #
    # print error


    #
    # print "start:"
    # data = input_data.read_data_sets("../data/MNIST/", one_hot=True)
    # print "data loaded"
    # training_features = data.train.images
    # training_labels = data.train.labels
    #
    # # Creating a list of features and labels:
    #
    # print training_features.shape
    # print training_labels.shape
    #
    # training_features = add_offset(training_features)
    # weights = optimize_weights(training_features,training_labels)
    #
    #







