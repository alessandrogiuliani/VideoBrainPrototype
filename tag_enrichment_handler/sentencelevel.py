# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:42:30 2020

@author: Alessandro Giuliani, Maria Madalina Stanciu

"""
import pandas as pd
import numpy as np
# from tagger import Tagger
import warnings
from scipy.spatial.distance import cosine, cityblock, canberra, euclidean, minkowski, braycurtis

warnings.filterwarnings('ignore')

TEMPLATE = ['Category',
            'videoUrl',
            'Tags_count',
            'google_trends',
            'cosine',
            'cityblock',
            'canberra_distance',
            'euclidean',
            'minkowski',
            'braycurtis',
            'similarity_word_2_vec']



def select_trends(tags, trends, num_trends, sentence_vectorizer):
    tags_df = pd.DataFrame(columns=TEMPLATE)
    keywords_emb = sentence_vectorizer.sent2vec(" ".join(tags))
    sentences=list(set(trends).difference(set(tags)))
    print (len(sentences), ' sentences ' )
    final_df = None
    for sentence in sentences:
        sentence_emb = sentence_vectorizer.sent2vec(sentence)
        if (len(sentence_emb) == 0) or (len(keywords_emb) == 0):
            continue
        similarity = sentence_vectorizer.get_cosine_similarity(keywords_emb, sentence_emb)
        cosine_distance = cosine(keywords_emb, sentence_emb)
        cityblock_distance = cityblock(keywords_emb, sentence_emb)
        canberra_distance = canberra(keywords_emb, sentence_emb)
        euclidean_distance = euclidean(keywords_emb, sentence_emb)
        minkowski_distance = minkowski(keywords_emb, sentence_emb, 3)
        braycurtis_distance = braycurtis(keywords_emb, sentence_emb)
        d = {'Category':'pass',
              'videoUrl':'pass',
              'Tags_count':len(tags),
              'google_trends':sentence,
              'cosine':cosine_distance,
              'cityblock':cityblock_distance,
              'canberra_distance':canberra_distance,
              'euclidean':euclidean_distance,
              'minkowski':minkowski_distance,
              'braycurtis':braycurtis_distance,
              'similarity_word_2_vec':similarity}
        tags_df.loc[len(tags_df)] = d
    __mask_lower = np.vectorize(lambda x, threshold: 1 if x < threshold else 0)
    dummy=tags_df.copy()
    dummy=dummy.loc[dummy['similarity_word_2_vec']>0].sort_values(['similarity_word_2_vec'], ascending=[False]).copy()
    dummy['label_cosine'] = 0
    dummy['label_cityblock'] = 0
    dummy['label_canberra_distance'] = 0
    dummy['label_euclidean'] = 0
    dummy['label_minkowski'] = 0
    dummy['label_braycurtis'] = 0
    #dummy['label_EMD'] = 0
    for row, group in dummy.sort_values(['similarity_word_2_vec']).groupby(['Category','videoUrl']):
        dummy.loc[group.index,'label_cosine']=1
        dummy.loc[group.index,'label_cityblock']=1
        dummy.loc[group.index,'label_canberra_distance']=1
        dummy.loc[group.index,'label_euclidean']=1
        dummy.loc[group.index,'label_minkowski']=1
        dummy.loc[group.index,'label_braycurtis']=1
        #dummy.loc[group.index,'label_EMD']=__mask_lower(group['EMD'],group['EMD'].mean())
        final_df= dummy.sort_values(['similarity_word_2_vec'],ascending=[False])\
.loc[(dummy['label_cosine']==1)&(dummy['label_cityblock']==1)&(dummy['label_canberra_distance']==1)&(dummy['label_minkowski']==1)&(dummy['label_euclidean']==1)&(dummy['label_braycurtis']==1)]\
.groupby(['Category','videoUrl'])['videoUrl','similarity_word_2_vec','google_trends'].head(num_trends)\
.reset_index()
    if final_df is None:
        return None
    return list(final_df['google_trends'])


