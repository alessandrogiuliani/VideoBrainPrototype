import pandas as pd
import numpy as np
import nltk
from nltk.cluster import KMeansClusterer
# from tagger import Tagger
import warnings
from scipy.spatial.distance import cosine, cityblock, canberra, euclidean, minkowski, braycurtis

warnings.filterwarnings('ignore')


TEMPLATE = ['Tags',
            'cluster',
            'cluster_id',
            'Tags_count',
            'google_trends',
            'cosine',
            'cityblock',
            'canberra_distance',
            'euclidean',
            'minkowski',
            'braycurtis',
            'similarity_word_2_vec',
            'cosine_cl',
            'cityblock_cl',
            'canberra_distance_cl',
            'euclidean_cl',
            'minkowski_cl',
            'braycurtis_cl',
            'similarity_word_2_vec_cl']




def get_keywords(sentences):
    keywords=[]
    for s in sentences:
        for k in s.split():
            if (not k.isnumeric()):
                keywords.append(k)
    return set(keywords)




def select_trends(tags, trends, num_trends, sentence_vectorizer, n_cls = 3):
    keywords= [w for w in get_keywords(tags) if w in sentence_vectorizer.model.wv.vocab]
    ck=min([len(keywords),n_cls])
    X = sentence_vectorizer.model[keywords]
    kclusterer = KMeansClusterer(ck, distance = nltk.cluster.util.cosine_distance, repeats=25)
    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
    clusters = {}
    for i,w in zip(assigned_clusters, keywords):
        if i in clusters.keys():
            clusters[i].append(w)
        else:
            clusters[i] = [w]
    clusters_df = pd.DataFrame(columns=TEMPLATE)
    keywords_emb = sentence_vectorizer.sent2vec(" ".join(tags))
    sentences = list(set(trends).difference(set(tags)))
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
        for cl, words in clusters.items():
            cluster_sentence =' '.join(words)
            cluster_embed = sentence_vectorizer.sent2vec(cluster_sentence)
            if len(cluster_embed)==0:
              continue
            similarity_cl = sentence_vectorizer.get_cosine_similarity(cluster_embed, sentence_emb)
            cosine_distance_cl = cosine(cluster_embed, sentence_emb)
            cityblock_distance_cl = cityblock(cluster_embed, sentence_emb)
            canberra_distance_cl = canberra(cluster_embed, sentence_emb)
            euclidean_distance_cl = euclidean(cluster_embed, sentence_emb)
            minkowski_distance_cl = minkowski(cluster_embed, sentence_emb, 3)
            braycurtis_distance_cl = braycurtis(cluster_embed, sentence_emb)
            #store information about all the distances and cosine similarity
            clusters_df.loc[len(clusters_df)] = {'Tags': ','.join(tags),
                                                 'cluster':cluster_sentence,
                                                 'cluster_id':cl,
                                                 'Tags_count':len(keywords),
                                                 'google_trends':sentence,
                                                 'cosine_cl':cosine_distance_cl,
                                                 'cityblock_cl':cityblock_distance_cl,
                                                 'canberra_distance_cl':canberra_distance_cl,
                                                 'euclidean_cl':euclidean_distance_cl,
                                                 'minkowski_cl':minkowski_distance_cl,
                                                 'braycurtis_cl':braycurtis_distance_cl,
                                                 'similarity_word_2_vec_cl':similarity_cl,
                                                 'cosine':cosine_distance,
                                                 'cityblock':cityblock_distance,
                                                 'canberra_distance':canberra_distance,
                                                 'euclidean':euclidean_distance,
                                                 'minkowski':minkowski_distance,
                                                 'braycurtis':braycurtis_distance,
                                                 'similarity_word_2_vec':similarity}
    __mask_greater = np.vectorize(lambda x, threshold: 1 if x > threshold else 0)
    __mask_lower = np.vectorize(lambda x, threshold: 1 if x < threshold else 0)
    max_phrases=10;  
    dummy=clusters_df.copy()
    dummy=dummy.loc[(dummy['similarity_word_2_vec']>0) & (dummy['similarity_word_2_vec_cl']>0)].sort_values(['similarity_word_2_vec','similarity_word_2_vec_cl'],ascending=[False,False]).copy()
    dummy['label_cosine_cl'] = 0
    dummy['label_cityblock_cl'] = 0
    dummy['label_canberra_distance_cl'] = 0
    dummy['label_euclidean_cl'] = 0
    dummy['label_minkowski_cl'] = 0
    dummy['label_braycurtis_cl'] = 0
    for row, group in dummy.loc[dummy['similarity_word_2_vec']>0].sort_values(['similarity_word_2_vec']).groupby(['Tags','cluster_id']):
        dummy.loc[group.index,'label_cosine_cl']=__mask_lower(group['cosine_cl'],group['cosine_cl'].mean())
        dummy.loc[group.index,'label_cityblock_cl']=__mask_lower(group['cityblock_cl'],group['cityblock_cl'].mean())
        dummy.loc[group.index,'label_canberra_distance_cl']=__mask_lower(group['canberra_distance_cl'],group['canberra_distance_cl'].mean())
        dummy.loc[group.index,'label_euclidean_cl']=__mask_lower(group['euclidean_cl'],group['euclidean_cl'].mean())
        dummy.loc[group.index,'label_minkowski_cl']=__mask_lower(group['minkowski_cl'],group['minkowski_cl'].mean())
        dummy.loc[group.index,'label_braycurtis_cl']=__mask_lower(group['braycurtis_cl'],group['braycurtis_cl'].mean())
    intermediatedf= dummy.loc[dummy['similarity_word_2_vec']>0]\
.loc[(dummy['label_cosine_cl']==1)&(dummy['label_cityblock_cl']==1)&(dummy['label_canberra_distance_cl']==1)&(dummy['label_minkowski_cl']==1)&(dummy['label_euclidean_cl']==1)&(dummy['label_braycurtis_cl']==1)]\
.sort_values(['similarity_word_2_vec', 'similarity_word_2_vec_cl'],ascending=[False,False])\
.groupby(['Tags','google_trends']).nth(0)\
.reset_index()  
    final_df = intermediatedf.sort_values(['similarity_word_2_vec'],ascending=[False]).groupby(['Tags'])['Tags','cluster','similarity_word_2_vec','google_trends'].head(num_trends)\
.reset_index()
    return list(final_df['google_trends'])


