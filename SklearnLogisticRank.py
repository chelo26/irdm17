import numpy as np
from collections import defaultdict
import time
from __future__ import division
from sklearn import linear_model, datasets
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
import pandas as pd
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

def training(features,labels,learning_rate=0.001):

    # Normalizing features:
    #features = normalize_features(features)

    # Logistic regression  model:
    #logreg = linear_model.LogisticRegression(C=1.5)
    #logreg.fit(features, labels)
    class_w = get_class_weight(labels)
    # Logistic regression with SGD:
    logSGD=linear_model.SGDClassifier(loss="log",class_weight=class_w,alpha=learning_rate, n_iter=25)
    logSGD.fit(features,labels)
    print "model trained"
    return logSGD

def find_parameters(features,labels):
    # Normalizing features:

    #features = features[0:10000]
    #labels = labels[0:10000]
    parameters = {'alpha': [0.00001, 0.00001, 0.0001,0.0005, 0.001, 0.005,0.01, 0.05, 0.1]}
    class_w = get_class_weight(labels)
    model = linear_model.SGDClassifier(loss="log", class_weight=class_w,n_iter=25)
    clf = GridSearchCV(model, parameters)
    clf.fit(features, labels)

    print "model trained"
    return clf


def dcg_score(y_true, y_score, k=10, gains="exponential"):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best


def evaluate(model, dictio_val, mean_Xval, std_Xval):
    dictio_evaluation = defaultdict(list)

    for key in dictio_val.keys():
        temp_list = dictio_val[key]
        for features_vec, relevance in temp_list:
            # Features:
            features_norm = (np.array(features_vec) - mean_Xval) / std_Xval
            features_norm = features_norm.reshape(1,-1)
            #features_norm = features_norm.reshape(-1,1)

            # Prediction:
            prediction = model.predict(features_norm)

            # Dictionary:
            dictio_evaluation[key].append((prediction[0],relevance))

            # print features_vec,relevance

    return dictio_evaluation

def order_tuples(list_tuples):
    #list_tuples=sorted(list_tuples, reverse=True, key=lambda tup: (tup[1], tup[0]))
    list_tuples = sorted(list_tuples, reverse=True, key=lambda tup: (tup[1]))
    return list_tuples


def reorder_dictio(dictio_eval):
    for key in dictio_eval.keys():
        #dictio_eval[key]=sorted(dictio_eval[key], reverse=True,key=lambda tup: (tup[1], tup[0]))
        dictio_eval[key] = sorted(dictio_eval[key], reverse=True, key=lambda tup: tup[1])
    return dictio_eval

def ndcg(dictio_eval):
    ndcg=[]
    for key in dictio_eval.keys():
        y_true,y_pred=separate(dictio_eval[key])
        if len(y_true)>10:
            ndcg.append(ndcg_score(y_true,y_pred,5))
    return ndcg


def separate(relevance_tuple):
    y_true=[]
    y_pred = []
    for tup in relevance_tuple:
        y_pred.append(tup[0])
        y_true.append(tup[1])
    return y_true,y_pred

# Functions to calculate ERR:
def get_proba(list_tuples):
    list_proba=[]
    for r_pred,r_true in list_tuples:
        proba = (np.power(2,r_pred)-1)/np.power(2,4)
        list_proba.append(proba)
    return list_proba

def get_ERR(list_proba,n=10,gamma=0.5):
    r=2
    err = list_proba[0]
    last_proba=1
    for i in xrange(1,len(list_proba)):
        actual_proba=list_proba[i]
        previous_proba=(1-list_proba[i-1])*last_proba
        #print proba
        stop_proba=actual_proba*previous_proba
        err+=stop_proba/r
        last_proba=previous_proba
        r+=1
    return err


def ERR(dictio_eval,n=10,gamma=0.5):
    list_ERR=[]
    for key in dictio_eval.keys():
        list_tuples=dictio_eval[key]
        list_proba=get_proba(list_tuples)
        err_result=get_ERR(list_proba,n,gamma)
        list_ERR.append(err_result)
    return list_ERR






if __name__ == '__main__':
    time_start = time.clock()
    # Paths:
    fold=defaultdict(list)
    i=1
    #for i in xrange(1,6):

        # training_path='./MSLR-WEB10K/Fold1/train.txt'
        # validation_path='./MSLR-WEB10K/Fold1/vali.txt'
        # test_path = './MSLR-WEB10K/Fold1/test.txt'

    training_path = './MSLR-WEB10K/Fold'+str(i)+'/train.txt'
    validation_path = './MSLR-WEB10K/Fold'+str(i)+'/vali.txt'
    test_path = './MSLR-WEB10K/Fold'+str(i)+'/test.txt'

    #Read training data
    X_train,y_train, dictio_train= readDataset(training_path)
    X_train,_,_=normalize_features(X_train)

    # Training:
    time_start = time.clock()
    model=training(X_train,y_train,learning_rate=0.1)
    print "time: " + str(time.clock() - time_start)

    # Validation:
    val_features, y_val, dictio_val= readDataset(validation_path)
    X_val,mean_Xval,std_Xval=normalize_features(val_features)

    # Evaluation
    y_pred=model.predict(X_val)

    #print precision_score(y_pred,y_val,average="micro")

    dictio_eval = evaluate(model, dictio_val, mean_Xval, std_Xval)
    dictio_eval2 = reorder_dictio(dictio_eval)
    total_ndcg=ndcg(dictio_eval2)
    print "validation error NDCG: %0.4f" %np.nanmean(total_ndcg)


    total_err_val = ERR(dictio_eval2, 10)
    print "test error ERR val: %0.4f" % np.mean(total_err_val)

    # Test set:
    test_features, y_test, dictio_test= readDataset(test_path)
    X_test,mean_Xtest,std_Xtest=normalize_features(test_features)

    # Evaluation
    y_pred_test=model.predict(X_test)

    print precision_score(y_pred_test,y_test,average="micro")

    dictio_eval_test = evaluate(model, dictio_test, mean_Xtest, std_Xtest)
    dictio_eval2_test = reorder_dictio(dictio_eval_test)
    total_ndcg_test=ndcg(dictio_eval2_test)
    print "test error NDCG test : %0.4f"%np.nanmean(total_ndcg_test)

    # Find Parameters:
    #grid_search_results=find_parameters(X_train,y_train)


    # Test set:

    total_err_test = ERR(dictio_eval2_test, 10)
    print "test error ERR test: %0.4f" % np.mean(total_err_test)

    fold[i].append((np.nanmean(total_ndcg),np.mean(total_err_val),np.nanmean(total_ndcg_test),np.mean(total_err_test)))


    print "time: "+str(time.clock()-time_start)

    #results=pd.DataFrame.from_dict(fold)
    #results.to_csv("results_LR.csv")