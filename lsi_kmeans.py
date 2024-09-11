#lsi_kmeans

from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
from gensim import corpora, models, similarities
import pickle
import numpy as np
import datetime
import time
import timeit
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


result=open(r'E:\\result\\asist2017\\lsi_kmeans.txt','a')
filelist=['chall','enall']
clusterlist=[25]#10,20,30,40,50
for files in filelist:
    dictionary = corpora.Dictionary.load('E:\\deal\\asist2017\\'+files+'\\'+files+'.dict')
    corpus = corpora.MmCorpus('E:\\deal\\asist2017\\'+files+'\\'+files+'.mm')
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    for i in range(20):
        s=10*(i+1)
        #LSIÄ£ÐÍ
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=s)
        lsi_vec=lsi[corpus_tfidf]   
        features=[]
        for term in lsi_vec:
            features.append([terms[1] for terms in term])
        for clusters in clusterlist:
            for m in range(10):
                kms = KMeans(init='k-means++', n_clusters=clusters).fit(np.array(features)) 
                labels = kms.labels_
                true_labels=open('E:\\deal\\asist2017\\label.txt').read().split('\n')   
                #ss=metrics.silhouette_score(np.array(features), labels, metric='sqeuclidean')
                #homogeneity_score=metrics.homogeneity_score(true_labels,labels)
                #completeness_score=metrics.completeness_score(true_labels,labels)
                v_measure_score=metrics.v_measure_score(true_labels,labels)
                result.write(str(files)+' '+str(s)+' '+str(v_measure_score)+'\n')
                             #str(ss)+' '+
                             #str(homogeneity_score)+' '+str(completeness_score)+' '+
                print files,' lsi dimension',s,' is done'
result.close()



        
