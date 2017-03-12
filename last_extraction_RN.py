import numpy as np
from collections import defaultdict
import time


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
    print('Reading training data from file...')
    with open(path, 'r') as file:
        #k=0
        features_list=[]
        rank_list=[]
        query_list=[]
        for line in file:
            split = line.split()
            features_list.append(extractFeatures(split))
            rank_list.append(int(split[0]))
            query_list.append(extractQueryData(split))
            #k+=1
            #if k==100:
            #    break
    print('Number of query ID %d' %(len(features_list)))
    return features_list,rank_list,query_list


# Normalisation:
def normalize_features(features):
    features=np.array(features)

    # Substracting the mean:
    mean_features = np.mean(features, axis=0)
    features = features - mean_features

    # Dividing by the std:
    std_features = np.std(features, axis=0)
    features = features / std_features
    print "features normalized"
    return features


# We put everything in a dictionary (key,value)= (query_id,[features,rank])
def make_dictionary(features,ranks,queries):
    dictio_quid=defaultdict(list)
    for feature_vec,rank,query in zip(features,ranks,queries):
        dictio_quid[query].append((feature_vec, rank))
    return dictio_quid

# Given a query ID, we separate on: [Xi,Xj,P_true] where P_true is either 0,0.5 or 1
def get_pairs_features(dictio_quid_featsRank):
    data = []
    k = 0
    for key in dictio_quid_featsRank.keys():
        # Temporary list of features,rank
        temp_list = dictio_quid_featsRank[key]

        for i in xrange(0, len(temp_list)):
            X1 = temp_list[i][0]
            rank1 = temp_list[i][1]
            for j in xrange(i + 1, len(temp_list)):
                X2 = temp_list[j][0]
                rank2 = temp_list[j][1]

                # Only look at queries with different id:
                if (rank1 == rank2):
                    break
                # data.append((X1,X2,0.5))
                if (rank1 > rank2):
                    data.append((X1, X2, int(1)))
                else:
                    data.append((X1, X2, int(0)))
        k += 1
        if k % 100 == 0:
            print "number of keys transformed: %d finished" % int(k)
    return data


# Putting in the good format for tensorflow:


def separate(data):
    Xi = []
    Xj = []
    P_target = []
    for instance in data:
        Xi.append(instance[0])
        Xj.append(instance[1])
        P_target.append(instance[2])
    return (np.array(Xi), np.array(Xj), np.array(P_target))

# Sampling:
def sampling_data(training_data, batch_size):
    N = len(training_data)
    indices = np.random.choice(N, batch_size)
    print "%d indices Selected" % batch_size
    return [training_data[i] for i in indices]



if __name__ == '__main__':
    time_start = time.clock()

    #Read training data
    features,ranks,queries = readDataset('./MSLR-WEB10K/Fold1/train.txt')
    features=normalize_features(features)

    # Making a dictionary:
    dictio_quid = make_dictionary(features, ranks, queries)

    # Getting the paris of features vectors:
    training_data = get_pairs_features(dictio_quid)

    # Sampling:
    sampled_data = sampling_data(training_data,)

    # Separating into array to put in tensorflow
    Xi, Xj, P_target = separate(data)

    print "time: "+str(time.clock()-time_start)