# -*- coding: cp936 -*-
#d2v_birch

from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import Birch
from sklearn import metrics
from gensim import corpora, models, similarities
from scipy.sparse import csr_matrix, coo_matrix
import pickle
import numpy as np
import datetime
import time
import timeit
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

result=open(r'E:\\result\\asist2017\\d2v_birch.txt','a')
filelist=['chall','enall']
clusterlist=[25]
thresh=[0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055]
for files in filelist:
    for i in range(20):
        s=10*(i+1)
        #Doc2Vec initialize
        div_texts = []
        f = open('E:\\deal\\asist2017\\'+files+'\\'+files+'.txt')
        lines = f.readlines()
        f.close()
        for line in lines:
            div_texts.append(line.strip().split(" "))
        sentences = []
        for i in range(len(div_texts)):
            string = "DOC_" + str(i)
            sentence = models.doc2vec.LabeledSentence(div_texts[i], labels = [string])
            sentences.append(sentence)
        d2v = models.Doc2Vec(sentences, size = s, window = 5, min_count = 0, dm = 1)
        #Doc2Vec train
        for i in range(10):
            d2v.train(sentences)
        features=[]
        for i,term in enumerate(sentences):
            feature=[]
            string = "DOC_" + str(i)
            for term in d2v[string]:
                feature.append(term)
            features.append(feature)
        for clusters in clusterlist:
            for resh in thresh:
                brch = Birch(threshold=resh,n_clusters=clusters).fit(np.array(features))#threshold=0.01
                labels = brch.labels_
                true_labels=open('E:\\deal\\asist2017\\label.txt').read().split('\n')
                #homogeneity_score=metrics.homogeneity_score(true_labels,labels)
                #completeness_score=metrics.completeness_score(true_labels,labels)
                v_measure_score=metrics.v_measure_score(true_labels,labels)
                #ss=metrics.silhouette_score(np.array(features), labels, metric='sqeuclidean')
                result.write(str(files)+' '+str(resh)+' '+str(s)+' '+str(v_measure_score)+'\n')
                #str(ss)+' '+str(homogeneity_score)+' '+str(completeness_score)+' '+                             
                print files,' d2v dimension',s,' is done'
result.close()



