# -*- coding: cp936 -*-
#d2v_dbscan

from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn import metrics
from gensim import corpora, models, similarities
from scipy.sparse import csr_matrix, coo_matrix
import pickle
import numpy as np
import datetime
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

result=open(r'E:\\result\\asist2017\\d2v_dbscan.txt','a')
threshold=[1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45]
filelist=['chall','enall']
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
        for thre in threshold:
            db = DBSCAN(eps=thre,min_samples=2).fit(np.array(features))
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            print thre,' ',s,'  ',n_clusters_
            '''
            true_labels=open('E:\\deal\\asist2017\\label.txt').read().split('\n')
            #ss=metrics.silhouette_score(np.array(features),labels, metric='sqeuclidean')
            #homogeneity_score=metrics.homogeneity_score(true_labels,labels)
            #completeness_score=metrics.completeness_score(true_labels,labels)
            v_measure_score=metrics.v_measure_score(true_labels,labels)
            result.write(str(thre)+' '+str(files)+' '+str(s)+' '+str(n_clusters_)+' '+str(v_measure_score)+'\n')
                         #str(ss)+' '+str(homogeneity_score)+' '+str(completeness_score)+' '+
            print 'd2v dimension ',s,'  ',files,' is done'
            '''
result.close()

        
