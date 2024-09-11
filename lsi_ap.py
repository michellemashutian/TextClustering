#lsi_ap

from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from gensim import corpora, models, similarities
import pickle
import numpy as np
import logging
import time
import timeit
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

threshold=[-0.2,-0.4,-0.6,-0.8,-1.0,
           -1.2,-1.4,-1.6,-1.8,-2.0,
           -2.2,-2.4,-2.6,-2.8,-3.0]

result=open(r'E:\\result\\asist2017\\lsi_ap.txt','a')
filelist=['chall','enall']
for files in filelist:
    dictionary = corpora.Dictionary.load('E:\\deal\\asist2017\\'+files+'\\'+files+'.dict')
    corpus = corpora.MmCorpus('E:\\deal\\asist2017\\'+files+'\\'+files+'.mm')
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    for i in range(20):
        s=10*(i+1)
        #LSIÄ£ÐÍ median
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=s)
        lsi_vec=lsi[corpus_tfidf]   
        features=[]
        for term in lsi_vec:
            features.append([terms[1] for terms in term])
        for thre in threshold:
            af = AffinityPropagation(preference=thre).fit(np.array(features))
            labels = af.labels_
            cluster_centers_indices = af.cluster_centers_indices_
            n_clusters_ = len(cluster_centers_indices)
            print files,' dimension ',s,' ',n_clusters_,' ',thre
            true_labels=open('E:\\deal\\asist2017\\label.txt').read().strip().split('\n')
            #ss=metrics.silhouette_score(np.array(features), labels, metric='sqeuclidean')
            #homogeneity_score=metrics.homogeneity_score(true_labels,labels)
            #completeness_score=metrics.completeness_score(true_labels,labels)
            v_measure_score=metrics.v_measure_score(true_labels,labels)
            result.write(str(thre)+' '+str(files)+' '+str(s)+' '+str(n_clusters_)+' '+str(v_measure_score)+'\n')
                         #str(ss)+' '+
                         #str(homogeneity_score)+' '+str(completeness_score)+' '+
                         #str(v_measure_score)+' '+str(timeUse)+'\n')
            print files,' dimension ',s,' ',n_clusters_,' ',v_measure_score

result.close()




        
