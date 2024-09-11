#lda_birch
import math
import pickle
import lda
import numpy as np
import datetime
from sklearn.cluster import Birch
from sklearn import metrics
import string
import time
import timeit
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.feature_extraction.text import CountVectorizer

result=open(r'E:\\result\\asist2017\\lda_birch.txt','a')
filelist=['chall','enall']
clusterlist=[25]
thresh=[0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055]
for files in filelist:
    for i in range(20):
        s=10*(i+1)
        lda_vec=open('E:\\deal\\asist2017\\'+files+'\\lda\\'+files+'_lda%s.vec'%s,'r').read().split('\n')
        features=[]
        for term in lda_vec:
            features.append([string.atof(num) for num in term.split(' ')])
        for clusters in clusterlist:
            for resh in thresh:
                brch = Birch(threshold=resh,n_clusters=clusters).fit(np.array(features))
                labels = brch.labels_
                true_labels=open('E:\\deal\\asist2017\\label.txt').read().split('\n')   
                #ss=metrics.silhouette_score(np.array(features), labels, metric='sqeuclidean')
                #homogeneity_score=metrics.homogeneity_score(true_labels,labels)
                #completeness_score=metrics.completeness_score(true_labels,labels)
                v_measure_score=metrics.v_measure_score(true_labels,labels)
                result.write(str(files)+' '+str(resh)+' '+str(s)+' '+str(v_measure_score)+'\n')
                #str(ss)+' '+str(homogeneity_score)+' '+str(completeness_score)+' '+                            
                print files,' lda dimension',s,' is done'
result.close()




