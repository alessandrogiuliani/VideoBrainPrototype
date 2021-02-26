# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 17:48:51 2020

@author: vegex
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:25:26 2020

@author: vegex
"""


import pandas as pd
import numpy as np
# from tagger import Tagger
import warnings
from scipy.spatial.distance import cosine, cityblock, canberra, euclidean, minkowski, braycurtis

warnings.filterwarnings('ignore')

TEMPLATE = ['word',
                    'Tags_count',
                    'google_trends',
                    'cosine',
                    'cityblock',
                    'canberra_distance',
                    'euclidean',
                    'minkowski',
                    'braycurtis',
                    'similarity_word_2_vec',
                    'cosine_word',
                    'cityblock_word',
                    'canberra_distance_word',
                    'euclidean_word',
                    'minkowski_word',
                    'braycurtis_word',
                    'similarity_word_2_vec_word']



def weighting_match(x):
    d={}
    d['similarity_word_2_vec'] = x['similarity_word_2_vec'].mean()
    return pd.Series(d)




def select_trends(tags, trends, num_trends, sentence_vectorizer):
    model = sentence_vectorizer.model
    tags_df = pd.DataFrame(columns=TEMPLATE)
    keywords_emb = sentence_vectorizer.sent2vec(" ".join(tags))
    sentences=list(set(trends).difference(set(tags)))
    print (len(sentences), ' sentences ' )
    for sentence in sentences:
        sentence_emb = sentence_vectorizer.sent2vec(sentence)
        if len(sentence_emb) == 0:
            continue
        similarity = sentence_vectorizer.get_cosine_similarity(keywords_emb, sentence_emb)
        cosine_distance = cosine(keywords_emb, sentence_emb)
        cityblock_distance = cityblock(keywords_emb, sentence_emb)
        canberra_distance = canberra(keywords_emb, sentence_emb)
        euclidean_distance = euclidean(keywords_emb, sentence_emb)
        minkowski_distance = minkowski(keywords_emb, sentence_emb, 3)
        braycurtis_distance = braycurtis(keywords_emb, sentence_emb)
        for index, word in enumerate(tags):
            word_emb = sentence_vectorizer.sent2vec(word)
            if (len(word_emb) == 0):
                continue
            similarity_word = sentence_vectorizer.get_cosine_similarity(word_emb,sentence_emb)
            cosine_distance_word = cosine(word_emb, sentence_emb)
            cityblock_distance_word = cityblock(word_emb, sentence_emb)
            canberra_distance_word = canberra(word_emb, sentence_emb)
            euclidean_distance_word = euclidean(word_emb, sentence_emb)
            minkowski_distance_word = minkowski(word_emb, sentence_emb, 3)
            braycurtis_distance_word = braycurtis(word_emb, sentence_emb)
            d = {'word':word,
                                 'Tags_count':len(tags),
                                 'google_trends':sentence,
                                 'cosine_word':cosine_distance_word,
                                 'cityblock_word':cityblock_distance_word,
                                 'canberra_distance_word':canberra_distance_word,
                                 'euclidean_word':euclidean_distance_word,
                                 'minkowski_word':minkowski_distance_word,
                                 'braycurtis_word':braycurtis_distance_word,
                                 'similarity_word_2_vec_word':similarity_word,
                                  'cosine':cosine_distance,
                                  'cityblock':cityblock_distance,
                                  'canberra_distance':canberra_distance,
                                  'euclidean':euclidean_distance,
                                  'minkowski':minkowski_distance,
                                  'braycurtis':braycurtis_distance,
                                  'similarity_word_2_vec':similarity
                                  }
            tags_df.loc[len(tags_df)] = d
    __mask_lower = np.vectorize(lambda x, threshold: 1 if x < threshold else 0)
    all_df=pd.DataFrame()
    for key, group in tags_df.loc[((tags_df['similarity_word_2_vec_word']>0) & (tags_df['similarity_word_2_vec']>0))]\
.sort_values(['similarity_word_2_vec','similarity_word_2_vec_word'],ascending=[True,True])\
.groupby(['word']):
        grp = group.copy()
        grp['label_cosine'] = __mask_lower(grp['cosine_word'],grp['cosine_word'].mean())
        grp['label_cityblock'] = __mask_lower(grp['cityblock_word'],grp['cityblock_word'].mean())
        grp['label_canberra_distance'] = __mask_lower(grp['canberra_distance_word'],grp['canberra_distance_word'].mean())
        grp['label_euclidean'] =  __mask_lower(grp['euclidean_word'],grp['euclidean_word'].mean())
        grp['label_minkowski'] = __mask_lower(grp['minkowski_word'],grp['minkowski_word'].mean())
        grp['label_braycurtis'] = __mask_lower(grp['braycurtis_word'],grp['braycurtis_word'].mean())
        grp= grp.loc[(grp['label_euclidean']==1)&(grp['label_cosine']==1)&(grp['label_cityblock']==1)&(grp['label_canberra_distance']==1)&(grp['label_minkowski']==1)&(grp['label_braycurtis']==1)]
        all_df= pd.concat([all_df,grp],axis=0,ignore_index=True)
        intermediate_df = all_df.loc[((all_df['similarity_word_2_vec_word']>0) & (all_df['similarity_word_2_vec']>0))].sort_values(['similarity_word_2_vec'],ascending=[False])\
.groupby(['google_trends'])\
.apply(weighting_match).reset_index()
        intermediate_df = intermediate_df.sort_values(['similarity_word_2_vec'],ascending=[False]).head(num_trends)
    return intermediate_df[['google_trends', 'similarity_word_2_vec']], all_df


