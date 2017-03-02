# Preprocessing script for data from MSLR (https://www.microsoft.com/en-us/research/project/mslr/)

import math
import numpy as np
import os
from collections import defaultdict
from collections import Counter
import tensorflow as tf 
from sklearn import preprocessing

def MSLR_load_format(data_dir, data_name):
    # Load, format and save MSLR data
    # Returns data as np.array, col1= rel, col2= qid, col3= feat1...col138= feat136
    
    print('Loading data')
    file = open(data_dir+ data_name, 'r')
    
    print('Formating data')
    raw_str_lst = file.read().split('\n')
    n_inst = len(raw_str_lst)
    print('Number of inst=', n_inst)
    data_lst_full = []
    for inst in raw_str_lst:
        str_lst = inst.split(' ')
        data_lst = []
        data_lst.append(str_lst[0])
        for dim in str_lst:
            tuples = dim.split(':')
            try:
                data_lst.append(tuples[1])
            except:
                pass
        data_lst_full.append(data_lst)
        if len(data_lst_full) % 10000 == 0:
            print((len(data_lst_full)/n_inst))
    data_array_full = np.asarray(data_lst_full)
    
    print('Loading complete. Data dim=', data_array_full.shape)
    print('Col1=  relevence, col2= qid, col3/col138= doc dims ')
    np.save('processed_array', data_array_full)

def select_qid(data, qid):
    # Select inst for specified qid, eg '116'
    
    data_sub = []
    for i in range(len(data)-1):
        if data[i][1] == qid:
            data_sub.append(data[i])
    print('Qid:', qid, np.asarray(data_sub).shape)
    return data_sub

def normalise_feats(data_sub):
    # Normalise features (per column) to between {0,1}
    
    min_max_scaler = preprocessing.MinMaxScaler()
    data_sub_norm = min_max_scaler.fit_transform(data_sub)
    return data_sub_norm

def generate_pairs(data_sub):
    # Generate pairs of non-matching relievence. Higher relevence 1st, lower 2nd
    
    pair_sets= []
    for i in data_sub:
        for j in data_sub:
            if int(i[0])> int(j[0]):
                pair_sets.append((i,j))
    print('#pairs:', np.asarray(pair_sets).shape)
    return pair_sets

def generate_pairs_opp(data_sub):
    # Generate pairs of non-matching relievence but opposite ordering. Lower relevence 1st, higher 2nd
    
    pair_sets= []
    for i in data_sub:
        for j in data_sub:
            if int(i[0])< int(j[0]):
                pair_sets.append((i,j))
                break
    print('#pairs:', np.asarray(pair_sets).shape)
    return pair_sets

#data_save = MSLR_import('/Users/jamesshields/MSc-Data-Science/IRDM/MSLR-WEB10K/Fold1/', 'train.txt')
#np.save('processed_array', data)
#data_load = np.load('processed_array.gz.npy')
#print(np.asarray(data_load).shape)
#data_subset = filter_for_qid(data_load, '166')
#data_sub_pairs = generate_pairs(data_subset)







