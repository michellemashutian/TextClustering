# -*- coding: cp936 -*-
#lsi_dbscan

from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn import metrics
from gensim import corpora, models, similarities
import pickle
import numpy as np
import time
import timeit
import logging
from sklearn.preprocessing import StandardScaler
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

result=open(r'E:\\result\\asist2017\\lsi_dbscan.txt','a')
threshold=[0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,
           0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30,0.31,
           0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.40,0.41,0.42,
           0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.50,0.51,0.52,0.53,0.54,0.55]
filelist=['chall','enall']
for files in filelist:
    dictionary = corpora.Dictionary.load('E:\\deal\\asist2017\\'+files+'\\'+files+'.dict')
    corpus = corpora.MmCorpus('E:\\deal\\asist2017\\'+files+'\\'+files+'.mm')
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    for i in range(20):
        s=10*(i+1)
        #LSI模型
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=s)
        lsi_vec=lsi[corpus_tfidf]
        features=[]
        for term in lsi_vec:
            features.append([terms[1] for terms in term])
        for thre in threshold:
            db = DBSCAN(eps=thre,min_samples=2).fit(np.array(features))
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            print thre,' ',s,'  ',n_clusters_
            true_labels=open('E:\\deal\\asist2017\\label.txt').read().split('\n')
            #ss=metrics.silhouette_score(np.array(features),labels, metric='sqeuclidean')
            #homogeneity_score=metrics.homogeneity_score(true_labels,labels)
            #completeness_score=metrics.completeness_score(true_labels,labels)
            v_measure_score=metrics.v_measure_score(true_labels,labels)
            result.write(str(files)+' '+str(thre)+' '+str(s)+' '+str(n_clusters_)+' '+str(v_measure_score)+'\n')
            #' '+str(ss)+' '+str(homogeneity_score)+' '+str(completeness_score)+' '+
            print 'lsi dimension ',s,' eps ',thre,' en is done'
result.close()




        
