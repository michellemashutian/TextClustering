#lda_ap
import math
import pickle
import lda
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import string
#import datetime
#import time
#import timeit
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.feature_extraction.text import CountVectorizer

threshold=[-0.2,-0.4,-0.6,-0.8,-1.0,
           -1.2,-1.4,-1.6,-1.8,-2.0,
           -2.2,-2.4,-2.6,-2.8,-3.0]
result=open(r'E:\\result\\asist2017\\lda_ap.txt','a')
filelist=['chall','enall']
for files in filelist:
    for i in range(20):
        s=10*(i+1)
        lda_vec=open('E:\\deal\\asist2017\\'+files+'\\lda\\'+files+'_lda%s.vec'%s,'r').read().strip().split('\n')
        features=[]
        for term in lda_vec:
            features.append([string.atof(num) for num in term.split(' ')])
        for thre in threshold:
            af = AffinityPropagation(preference=thre).fit(np.array(features))
            labels = af.labels_
            cluster_centers_indices = af.cluster_centers_indices_
            n_clusters_ = len(cluster_centers_indices)
            true_labels=open('E:\\deal\\asist2017\\label.txt').read().strip().split('\n')
            v_measure_score=metrics.v_measure_score(true_labels,labels)
            result.write(str(thre)+' '+str(files)+' '+str(s)+' '+str(n_clusters_)+' '+str(v_measure_score)+'\n')
            print files,' dimension ',s,' ',n_clusters_,' ',v_measure_score
result.close()

        

