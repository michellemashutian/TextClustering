#lda_kmeans
import math
import pickle
import lda
import numpy as np
import datetime
import time
import timeit
from sklearn.cluster import KMeans
from sklearn import metrics
import string
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.feature_extraction.text import CountVectorizer

result=open(r'E:\\result\\asist2017\\lda_kmeans.txt','a')
filelist=['chall','enall']
clusterlist=[25]
for files in filelist:
    for i in range(20):
        s=10*(i+1)
        lda_vec=open('E:\\deal\\asist2017\\'+files+'\\lda\\'+files+'_lda%s.vec'%s,'r').read().split('\n')
        features=[]
        for term in lda_vec:
            features.append([string.atof(num) for num in term.split(' ')])
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
                print files,' lda dimension',s,' is done'
result.close()


