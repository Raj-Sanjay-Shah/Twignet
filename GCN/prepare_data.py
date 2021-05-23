#!/usr/bin/python
#-*-coding:utf-8-*-

sentence_dir = '/Data/sentences.txt'
labels_dir = '/Data/labels.txt'
train_test_list = '/Data/train_or_test_list.txt'
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
parent_path = Path().resolve()

sentence_dir = str(parent_path) + sentence_dir
labels_dir = str(parent_path) + labels_dir
train_test_list = str(parent_path) + train_test_list

dataset_name = 'mr'

pickle_off = open (sentence_dir, "rb")
sentences = pickle.load(pickle_off)

pickle_off1 = open (labels_dir, "rb")
labels = pickle.load(pickle_off1)

pickle_off2 = open (train_test_list, "rb")
train_or_test_list = pickle.load(pickle_off2)

meta_data_list = []
print(sentences[0])
# print(sentences[8764],labels[8764],train_or_test_list[8764])
for i in range(len(sentences)):
    meta = str(i) + '\t' + train_or_test_list[i] + '\t' + labels[i]
    meta_data_list.append(meta)

meta_data_str = '\n'.join(meta_data_list)

f = open('GCN/data/' + dataset_name + '.txt', 'w+')
f.write(meta_data_str)
f.close()

corpus_str = '\n'.join(sentences)
f = open('GCN/data/corpus/' + dataset_name + '.txt', 'w+')
f.write(corpus_str)
f.close()
