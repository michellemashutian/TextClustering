#vsm_kmeans

from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
import pickle
import numpy as np
import datetime
import time
import timeit
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


result=open(r'E:\\result\\asist2017\\vsm_kmeans.txt','a')
filelist=['chall','enall']
clusterlist=[25]
for files in filelist:
    doc=open('E:\\deal\\asist2017\\'+files+'\\'+files+'.txt','r').read().split('\n')
    for i in range(60):
        s=500*(i+1)
        corpus_tfidf=TfidfVectorizer(min_df=1,max_features=s).fit_transform(doc)
        for clusters in clusterlist:
            for m in range(10):
                kms = KMeans(init='k-means++', n_clusters=clusters).fit(corpus_tfidf)
                labels = kms.labels_
                true_labels=open('E:\\deal\\asist2017\\label.txt').read().split('\n')
                v_measure_score=metrics.v_measure_score(true_labels,labels)
                result.write(str(files)+' '+str(s)+' '+str(v_measure_score)+'\n')
                print 'dimension ',s,' ',files,' is done'
'''
filelist=['chall','enall']
clusterlist=[10,20,30,40,50]
for files in filelist:
    doc=open('E:\\deal\\ttc\\'+files+'\\'+files+'.txt','r').read().decode('GBK').split('\n')
    for i in range(60):
        s=500*(i+1)
        corpus_tfidf=TfidfVectorizer(min_df=2,max_features=s).fit_transform(doc)
        for clusters in clusterlist:
            for m in range(10):
                #timeIn=time.clock()
                kms = KMeans(init='k-means++', n_clusters=clusters).fit(corpus_tfidf)
                #timeUse=time.clock()-timeIn
                #print 'Running time is ',timeUse
                labels = kms.labels_
                true_labels=open('D:\\deal\\catergory.txt').read().split('\n')
                ss=metrics.silhouette_score(corpus_tfidf.todense(), labels, metric='sqeuclidean')
                #homogeneity_score=metrics.homogeneity_score(true_labels,labels)
                #completeness_score=metrics.completeness_score(true_labels,labels)
                #v_measure_score=metrics.v_measure_score(true_labels,labels)
                result.write(str(clusters)+' '+str(files)+' '+str(s)+' '+
                             str(ss)+'\n')
                             #str(homogeneity_score)+' '+str(completeness_score)+' '+
                             #str(v_measure_score)+' '+str(timeUse)+'\n')
                print clusters,'clusters ',s,' ',files,' is done'

filelist=['zhmobile','zhwind']
clusterlist=[10,20,30,40,50]
for files in filelist:
    doc=open('D:\\deal\\ttc\\'+files+'\\'+files+'.txt','r').read().split('\n')
    for i in range(60):
        s=500*(i+1)
        corpus_tfidf=TfidfVectorizer(min_df=2,max_features=s).fit_transform(doc)
        for clusters in clusterlist:
            for m in range(10):
                #timeIn=time.clock()
                kms = KMeans(init='k-means++', n_clusters=clusters).fit(corpus_tfidf)
                #timeUse=time.clock()-timeIn
                #print 'Running time is ',timeUse
                labels = kms.labels_
                true_labels=open('D:\\deal\\catergory.txt').read().split('\n')
                ss=metrics.silhouette_score(corpus_tfidf.todense(), labels, metric='sqeuclidean')
                #homogeneity_score=metrics.homogeneity_score(true_labels,labels)
                #completeness_score=metrics.completeness_score(true_labels,labels)
                #v_measure_score=metrics.v_measure_score(true_labels,labels)
                result.write(str(clusters)+' '+str(files)+' '+str(s)+' '+
                             str(ss)+'\n')
                             #str(homogeneity_score)+' '+str(completeness_score)+' '+
                             #str(v_measure_score)+' '+str(timeUse)+'\n')
                print clusters,'clusters ',s,' ',files,' is done' 
'''
result.close()



        
