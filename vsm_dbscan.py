# -*- coding: cp936 -*-
#vsm_dbscan

from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn import metrics
import pickle#用于将训练好的tfidf对象保存到文件中，避免重复训练
import numpy as np
import time
import timeit
from sklearn.preprocessing import StandardScaler
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

result=open(r'E:\\result\\asist2017\\vsm_dbscan.txt','a')
filelist=['chall','enall']
threshold1=[0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,
            0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,
            0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,
            0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8]

for files in filelist:
    doc=open('E:\\deal\\asist2017\\'+files+'\\'+files+'.txt','r').read().split('\n')
    for i in range(60):
        s=500*(i+1)
        corpus_tfidf=TfidfVectorizer(min_df=1,max_features=s).fit_transform(doc)
        for thre in threshold1:
            db = DBSCAN(eps=thre,min_samples=2).fit(corpus_tfidf)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            if ((n_clusters_==0) or (n_clusters_==1)):
                continue
            else:
                true_labels=open('E:\\deal\\asist2017\\label.txt').read().split('\n')
                v_measure_score=metrics.v_measure_score(true_labels,labels)
                result.write(str(thre)+' '+str(files)+' '+str(s)+' '+str(n_clusters_)+' '+str(v_measure_score)+'\n')
                print files,' ',s,' ',thre,' is done'
'''
filelist=['zhmobile']
#threshold1=[1.26,1.27,1.28,1.29,1.3,1.31,1.32,1.33,1.34,1.35]
for files in filelist:
    doc=open('D:\\deal\\ttc\\'+files+'\\'+files+'.txt','r').read().split('\n')#.decode('GBK')
    for i in range(60):
        s=500*(i+1)
        corpus_tfidf=TfidfVectorizer(min_df=2,max_features=s).fit_transform(doc)
        for thre in threshold1:
            #timeIn=time.clock()
            db = DBSCAN(eps=thre,min_samples=2).fit(corpus_tfidf)
            #timeUse=time.clock()-timeIn
            #print 'Running time is ',timeUse
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            if ((n_clusters_==0) or (n_clusters_==1)):
                continue
            else:
                #true_labels=open('D:\\deal\\catergory.txt').read().split('\n')
                ss=metrics.silhouette_score(corpus_tfidf.todense(), labels, metric='sqeuclidean')
                #homogeneity_score=metrics.homogeneity_score(true_labels,labels)
                #completeness_score=metrics.completeness_score(true_labels,labels)
                #v_measure_score=metrics.v_measure_score(true_labels,labels)
                result.write(str(thre)+' '+str(files)+' '+str(s)+' '+str(n_clusters_)+' '+
                             str(ss)+'\n')
                             #str(homogeneity_score)+' '+str(completeness_score)+' '+
                             #str(v_measure_score)+' '+str(timeUse)+'\n')
                print files,' is done',s

filelist=['ch4904']#'ch','entoch','ch4904'
threshold2=[1.26,1.27,1.28,1.29,1.3,1.31,1.32,1.33,1.34,1.35]
for files in filelist:
    doc=open('D:\\deal\\'+files+'\\'+files+'.txt','r').read().decode('GBK').split('\n')
    corpus_tfidf=TfidfVectorizer().fit_transform(doc)
    for thre in threshold2:
        timeIn=time.clock()
        db = DBSCAN(eps=thre,min_samples=2).fit(corpus_tfidf)
        timeUse=time.clock()-timeIn
        print 'Running time is ',timeUse
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        true_labels=open('D:\\deal\\catergory.txt').read().split('\n')
        ss=metrics.silhouette_score(corpus_tfidf.todense(), labels, metric='sqeuclidean')
        homogeneity_score=metrics.homogeneity_score(true_labels,labels)
        completeness_score=metrics.completeness_score(true_labels,labels)
        v_measure_score=metrics.v_measure_score(true_labels,labels)
        result.write(str(thre)+' '+str(files)+' '+str(n_clusters_)+' '+
                     str(ss)+' '+
                     str(homogeneity_score)+' '+str(completeness_score)+' '+
                     str(v_measure_score)+' '+str(timeUse)+'\n')
        print files,' is done'
'''
result.close()




        
