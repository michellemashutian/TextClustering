#d2v_ap

from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from gensim import corpora, models, similarities
from scipy.sparse import csr_matrix, coo_matrix
import pickle
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

threshold=[-5,-10,-15,-20,-25,-30,-35,-40,-45,-50,-55]
result=open(r'E:\\result\\asist2017\\d2v_ap.txt','a')

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
            af = AffinityPropagation(preference=thre).fit(np.array(features))
            labels = af.labels_
            cluster_centers_indices = af.cluster_centers_indices_
            n_clusters_ = len(cluster_centers_indices)
            true_labels=open('E:\\deal\\asist2017\\label.txt').read().strip().split('\n')    
            #ss=metrics.silhouette_score(np.array(features), labels, metric='sqeuclidean')
            #homogeneity_score=metrics.homogeneity_score(true_labels,labels)
            #completeness_score=metrics.completeness_score(true_labels,labels)
            v_measure_score=metrics.v_measure_score(true_labels,labels)
            result.write(str(thre)+' '+str(files)+' '+str(s)+' '+
                         str(n_clusters_)+' '+str(v_measure_score)+'\n')
                         #str(ss)+' '+
                         #str(homogeneity_score)+' '+str(completeness_score)+' '++' '+str(timeUse)                        
            print 'd2v dimension ',s,'   ',files,'  ',thre,' ',n_clusters_,'  is done!'
result.close()

       
