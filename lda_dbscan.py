#lda_dbscan
import math
import pickle
import lda
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import string
import time
import timeit
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.feature_extraction.text import CountVectorizer

result=open(r'E:\\result\\asist2017\\lda_dbscan.txt','a')
threshold=[0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,
           0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30,0.31,
           0.32,0.33,0.34,0.35]
filelist=['chall','enall']
for files in filelist:
    for i in range(20):
        s=10*(i+1)
        lda_vec=open('E:\\deal\\asist2017\\'+files+'\\lda\\'+files+'_lda%s.vec'%s,'r').read().split('\n')
        features=[]
        for term in lda_vec:
            features.append([string.atof(num) for num in term.split(' ')])
        for thre in threshold:
            db = DBSCAN(eps=thre,min_samples=2).fit(np.array(features))
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            true_labels=open('E:\\deal\\asist2017\\label.txt').read().split('\n')
            #ss=metrics.silhouette_score(np.array(features),labels, metric='sqeuclidean')
            #homogeneity_score=metrics.homogeneity_score(true_labels,labels)
            #completeness_score=metrics.completeness_score(true_labels,labels)
            v_measure_score=metrics.v_measure_score(true_labels,labels)
            result.write(str(files)+' '+str(thre)+' '+str(s)+' '+str(n_clusters_)+' '+str(v_measure_score)+'\n')
                         #str(ss)+' '+str(homogeneity_score)+' '+str(completeness_score)+' '+
            print files,'  ',s,' ',thre,' ',n_clusters_
result.close()


