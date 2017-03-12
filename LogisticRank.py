import numpy as np
from collections import defaultdict
import time
from __future__ import division
from sklearn import linear_model, datasets
from sklearn.metrics import *

# Functions to extrac the documents, query and rank information
def extractFeatures(split):
    features = []
    for i in xrange(1, 138):
        features.append(float(split[i].split(':')[1]))
    # Convert to tuples:
    return features

def readDataset(path):
    #dictio_quid= defaultdict(list)
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

            # Getting rank:
            rank = int(split[0])
            rank_list.append(rank)

            k+=1
            #if k==1000:
            #   break
    print('Number of query ID %d' %(len(features_list)))
    return np.array(features_list),np.array(rank_list)

# Normalization:

def normalize_features(features):
    # Substracting the mean:
    mean_features=np.mean(features,axis=0)
    features=features-mean_features

    # Dividing by the std:
    std_features=np.std(features,axis=0)
    features=features/std_features
    print "features normalized"
    return features

def training(features,labels):

    # Normalizing features:
    #features = normalize_features(features)

    # Logistic regression  model:
    #logreg = linear_model.LogisticRegression(C=1.5)
    #logreg.fit(features, labels)

    # Logistic regression with SGD:
    logSGD=linear_model.SGDClassifier(loss="log")
    logSGD.fit(features,labels)
    print "model trained"
    return logSGD




if __name__ == '__main__':
    time_start = time.clock()
    # Paths:
    training_path='./MSLR-WEB10K/Fold1/train.txt'
    validation_path='./MSLR-WEB10K/Fold1/vali.txt'

    #Read training data
    X_train,y_train= readDataset(training_path)
    X_train=normalize_features(X_train)

    # Training:
    model=training(X_train,y_train)

    # Validation:
    val_features, y_val= readDataset(validation_path)
    X_val=normalize_features(val_features)

    # Evaluation
    y_pred=model.predict(X_val)

    print precision_score(y_pred,y_val,average="weighted")





    print "time: "+str(time.clock()-time_start)