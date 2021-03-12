# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:42:30 2020


@credits: Maria Madalina Stanciu
@author: Maria Madalina Stanciu, Alessandro Giuliani

"""
from pandas.io.json import json_normalize
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from datetime import date
from numpy.testing import assert_almost_equal
import time
import os
import numpy as np
# from tagger import Tagger
from gensim.models import Word2Vec,KeyedVectors
import nltk
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec,KeyedVectors
from gensim.models.wrappers import FastText
from nltk.cluster import KMeansClusterer
from urllib.request import urlopen
from bs4 import BeautifulSoup
import warnings
from google_trends_wrapper import utils
from tag_enrichment_handler import wordlevel, sentencelevel, clusterlevel
from nltk.corpus import abc
from nltk.stem import WordNetLemmatizer

#*****************************************************************************
#***********************  Global initialization   ****************************
#*****************************************************************************
warnings.filterwarnings('ignore')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('abc')
# #EMBEDDING_FILE = f'{os.getcwd()}/model_data/GoogleNews-vectors-negative300.bin.gz'


# #EMBEDDING_FILE = f'{os.getcwd()}/model_data/IT.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt'



# WORD2VEC = FastText.load_fasttext_format(f'{os.getcwd()}/model_data/it')
# print('Emebdding vectors loaded')
# #WORD2VEC = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=False, encoding='latin-1')
# #WORD2VEC = Word2Vec(abc.sents())
# # check this https://radimrehurek.com/gensim/models/word2vec.html
# # binary = true is because the word2vec file we have has a bin suffix, if it's text file, binary = false
# # key is the word, value is a 300 dimensional vector for each word
# WORD2VEC.init_sims(replace=True)



#*****************************************************************************
#************************   Utility functions    *****************************
#*****************************************************************************

#given a list of tags (formed of multiple words) the function outputs a maximum ``count'' tags and removes the numbers from tags
def get_tags_sentence(tags, word2vec=None, count=None):
  tags_v = []
  for tag in tags:
    t = ["".join(tg) for tg in tag.split(' ') if not tg.isnumeric()]
    if word2vec is not None:
      t = [w for w in t if w in word2vec.wv.vocab]
    if len(t):
      tags_v.append(" ".join(t))
  if (count):
      return tags_v[:count]
  return tags_v



#cosine similarity between two vectors
def get_cosine_similarity(feature_vec_1, feature_vec_2):
  return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]




#*****************************************************************************
#***********************    Class MySentence   *******************************
#*****************************************************************************

class MySentences(object):

    wnl = WordNetLemmatizer()

    def __init__(self, model, stop_words='english', coco_categories:pd.DataFrame=None):
        self.model = model
        self.coco_categories = coco_categories
        self.stop_words = stopwords.words(stop_words)
    

    
  #Transforms multiple words (sentence) into a single word embedding'''
    def sentence_vectorize(self,sentence):
        words = [word for word in sentence if word in self.model.wv.vocab and word not in self.stop_words]
        if len(words) >= 1:
            return np.mean(self.model[words], axis=0)
        else:
            return []



    #same as the one above. It is just computed differently
    def sent2vec(self, sentence):
        s=sentence.split()
        s = [self.wnl.lemmatize(wrd) for wrd in s]
        words = [w for w in s if w in self.model.wv.vocab and w not in self.stop_words]
        words = [w for w in words if w.isalpha()]
        if len(words) ==0:
            return []
        M = []
        for w in words:
            try:
                M.append(self.model[w])
            except:
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        return v / np.sqrt((v ** 2).sum())

    #computes the cosine similarity between two word embeddings
    def get_cosine_similarity(self, feature_vec_1, feature_vec_2):
        return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]

#*****************************************************************************
#*****************************************************************************
#*****************************************************************************


#*****************************************************************************
#*******************    Class YouTubeMetaExtractor   *************************
#*****************************************************************************

class YouTubeMetaExtractor(object):
    '''
    kwargs parameters:
        
        verbose (default True): print status messages during the execution.
        
        storeRelatedVideos (default True): set if load related videos properties.
        
    '''
    
    
    def __init__(self, ytURL, **kwargs):
        self.videoURLToken = 'https://www.youtube.com'
        self.searchURLToken = 'https://www.youtube.com/results?search_query='
        self.load_settings(ytURL, **kwargs)
        
        
        
    
    def load_settings(self, ytURL, **kwargs):
        self.verbose = kwargs.get('verbose', True)
        if self.verbose: print('Setting system')
        self.url = ytURL
        self.videoID = ytURL.replace('https://www.youtube.com/watch?v=', '') 
        self.storeRelatedVideos = kwargs.get('storeRelatedVideos', True)
        self.properties = self.get_metaProperties(ytURL)
        #self.get_relatedTags(self.get_relatedSearchLinks())
        
        
        
        
    def get_metaProperties(self, link, get_title = True):
        properties = dict()
        webpage = urlopen(link).read()
        soup = BeautifulSoup(webpage, features="lxml", from_encoding="iso-8859-1")
        tags = soup.find_all("meta",  property="og:video:tag")
        if get_title:
            title = soup.find_all("meta",  property="og:title")[0]['content']
            properties['title'] = title
        properties['tags'] = [x['content'] for x in tags]
        return properties
        




    def get_relatedSearchLinks(self):
        queryParameter = '+'.join(self.properties['title'].split()).encode('utf-8')
        relatedSearch = f'{self.searchURLToken}{queryParameter}'
        print(relatedSearch)
        temp = urlopen(relatedSearch)
        webpage = temp.read()
        soup = BeautifulSoup(webpage, 'lxml')
        data = soup.find_all("a", class_ = "yt-uix-tile-link yt-ui-ellipsis yt-ui-ellipsis-2 yt-uix-sessionlink spf-link")
        relatedUrls= [(self.videoURLToken + x.get('href')) for x in data if x.get('href') not in relatedSearch]
        return relatedUrls
        


    def get_tags(self, link):
        webpage = urlopen(link).read()
        soup = BeautifulSoup(webpage, "lxml")
        tags = soup.find_all("meta",  property="og:video:tag")
        return [x['content'] for x in tags]




    def get_relatedTags(self, relatedLinks):
        relatedTags = {k: [self.get_metaProperties(k, get_title = False)] for k in relatedLinks}
        self.properties['relatedtags'] = relatedTags


#*****************************************************************************
#*****************************************************************************
#*****************************************************************************



#*****************************************************************************
#**********************    Class TagGenerator   ******************************
#*****************************************************************************

class TagGenerator(object):


    catIds = utils.CategoryRead()
    selectors = {'WL' : wordlevel,
                 'SL' : sentencelevel,
                 'CL' : clusterlevel}
    languages = {'english': 'US',
                 'italian': 'IT'}
    mapping_category = {'food': catIds.get_category_id('Food & Drink'),
                        'cars': catIds.get_category_id('Autos & Vehicles'),
                        'animals': catIds.get_category_id('Pets & Animals'),
                        'tech': catIds.get_category_id('Computers & Electronics'),
                        'music': catIds.get_category_id('Music & Audio'),
                        'sport': catIds.get_category_id('Sports')}




    def __init__(self, model=None, **kwargs):
        language = kwargs.get('language', 'english')
        self.model =  model
        self.vectorizer= MySentences(self.model, stop_words=language)
        self.domain = kwargs.get('domain', None)
        if self.domain not in self.mapping_category.keys():
            print('Error: no valid domain selected')
        self.Gcategory = self.mapping_category[self.domain]
        granularity = kwargs.get('granularity', 'WL')
        self.selector = self.selectors[granularity]
        self.n_trends = kwargs.get('n_suggested_tags', 5)
        self.google_trends = {}
        self.catIds = utils.CategoryRead(geo=self.languages[language])




    def setGranularity(self, granularity):
        self.granularity = granularity




    def setLanguage(self, language):
        self.catIds = utils.CategoryRead(geo=self.languages[language])




    def setDomain(self, domain):
        self.domain = domain




    def getCandidateTrends(self):
        self.candidate_trends = self.catIds.set_related_searches(self.domain, self.Gcategory)




    def getTags(self, videoId):
        videoURL = f'https://www.youtube.com/watch?v={videoId}'
        yt_handler = YouTubeMetaExtractor(videoURL)
        originalTags = yt_handler.get_tags(videoURL)
        parsedOriginalTags = get_tags_sentence(originalTags, word2vec=self.model)
        self.getCandidateTrends()
        if len(parsedOriginalTags) == 0:
            return []
        return self.selector.select_trends(parsedOriginalTags,
                          self.candidate_trends,
                          self.n_trends,
                          self.vectorizer)



