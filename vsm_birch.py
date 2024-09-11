# -*- coding: cp936 -*-
#vsm_birch

from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import Birch
from sklearn import metrics
import pickle
import time
import timeit
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import datetime
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



'''
filelist=['en4904']#'en','chtoen',
clusterlist=[10,20,30,40,50]#10,20,
thre=[0.02,0.03,0.04,0.05]
for files in filelist:
    doc=open('D:\\deal\\'+files+'\\'+files+'.txt','r').read().split('\n')
    for i in range(20):
        s=10*(i+1)
        corpus_tfidf=TfidfVectorizer(min_df=2,max_features=s).fit_transform(doc)
        for clusters in clusterlist:
            timeIn=time.clock()
            for resh in thre:
                brch = Birch(threshold=resh,n_clusters=clusters).fit(corpus_tfidf)#
                timeUse=time.clock()-timeIn
                print 'Running time is ',timeUse
                labels = brch.labels_
                true_labels=open('D:\\deal\\catergory.txt').read().split('\n')
                ss=metrics.silhouette_score(corpus_tfidf, labels)
                homogeneity_score=metrics.homogeneity_score(true_labels,labels)
                completeness_score=metrics.completeness_score(true_labels,labels)
                v_measure_score=metrics.v_measure_score(true_labels,labels)
                result.write(str(clusters)+' '+str(files)+' '+str(s)+' '+
                             str(ss)+' '+
                             str(homogeneity_score)+' '+str(completeness_score)+' '+
                             str(v_measure_score)+' '+str(timeUse)+' '+str(resh)+'\n')
                print clusters,'clusters ',s,' ',files,' is done'
'''
result=open(r'E:\\result\\asist2017\\vsm_birch.txt','a')
filelist=['chall']#'en','chtoen','enmobile','enwind',,'enall'
clusterlist=[25]#10,20,
thre=[0.055]#0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,
for files in filelist:
    doc=open('E:\\deal\\asist2017\\'+files+'\\'+files+'.txt','r').read().split('\n')#.decode('GBK')
    for i in range(1):
        s=500*(i+60)
        corpus_tfidf=TfidfVectorizer(min_df=2,max_features=s).fit_transform(doc)
        for clusters in clusterlist:
            for resh in thre:
                brch = Birch(threshold=resh,n_clusters=clusters).fit(corpus_tfidf)
                labels = brch.labels_
                true_labels=open('E:\\deal\\asist2017\\label.txt').read().split('\n')
                v_measure_score=metrics.v_measure_score(true_labels,labels)
                result.write(str(files)+' '+str(s)+' '+str(resh)+' '+str(v_measure_score)+'\n')
                             #str(homogeneity_score)+' '+str(completeness_score)+' '+
                             #str(v_measure_score)+' '+str(timeUse)+' '+str(resh)+'\n')
                print 'dimension ',s,' ',files,' is done'
            
'''
filelist=['ch4904']#'ch','entoch','ch4904'
clusterlist=[30,40]#10,20,30,40,50
thre=[0.02,0.03,0.04,0.05]#0.02,0.03,0.04,0.05
for files in filelist:
    doc=open('D:\\deal\\'+files+'\\'+files+'.txt','r').read().decode('GBK').split('\n')
    corpus_tfidf=TfidfVectorizer().fit_transform(doc)
    for clusters in clusterlist:
        for resh in thre:
            timeIn=time.clock()
            brch = Birch(threshold=resh,n_clusters=clusters).fit(corpus_tfidf)#threshold=0.01,
            timeUse=time.clock()-timeIn
            print 'Running time is ',timeUse
            labels = brch.labels_
            true_labels=open('D:\\deal\\catergory.txt').read().split('\n')
            ss=metrics.silhouette_score(corpus_tfidf, labels)
            homogeneity_score=metrics.homogeneity_score(true_labels,labels)
            completeness_score=metrics.completeness_score(true_labels,labels)
            v_measure_score=metrics.v_measure_score(true_labels,labels)
            result.write(str(clusters)+' '+str(files)+' '+
                         str(ss)+' '+
                         str(homogeneity_score)+' '+str(completeness_score)+' '+
                         str(v_measure_score)+' '+str(timeUse)+' '+str(resh)+'\n')
            print clusters,'clusters ',files,' is done'
'''
result.close()




        
