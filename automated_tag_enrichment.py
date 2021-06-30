# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:42:30 2020

@author: Alessandro Giuliani, Maria Madalina Stanciu

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
from nltk.corpus import abc
from nltk.cluster import KMeansClusterer
from urllib.request import urlopen
from bs4 import BeautifulSoup
import warnings
from google_trends_wrapper import utils
from tag_enrichment_handler import wordlevel, sentencelevel, clusterlevel
from nltk.stem import WordNetLemmatizer
import pafy
from textblob import TextBlob
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
        self.get_title = kwargs.get('get_title', True)
        self.get_description = kwargs.get('get_description', True)
        self.get_original_tags = kwargs.get('get_original_tags', True)
        self.load_settings(ytURL, **kwargs)
        
        
        
    
    def load_settings(self, ytURL, **kwargs):
        self.verbose = kwargs.get('verbose', True)
        if self.verbose: print('Setting system')
        self.url = ytURL
        self.videoID = ytURL.replace('https://www.youtube.com/watch?v=', '') 
        self.storeRelatedVideos = kwargs.get('storeRelatedVideos', True)
        self.properties = self.get_metaProperties(ytURL)
        #self.get_relatedTags(self.get_relatedSearchLinks())
        
        
        
        
    def get_metaProperties(self, link):
        properties = list()
        webpage = urlopen(link).read()
        soup = BeautifulSoup(webpage, features="lxml", from_encoding="iso-8859-1")
        if self.get_original_tags:
            tags = soup.find_all("meta",  property="og:video:tag")
            properties += [x['content'] for x in tags]
        if self.get_title:
            title = soup.find_all("meta",  property="og:title")[0]['content']
            properties.append(title)           
        if self.get_description:
            description = soup.find_all("meta",  property="og:description")[0]['content']
            properties.append(description)
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

    def noun_tokenizer(self, string):
        tokenized = nltk.word_tokenize(string)
        is_noun = lambda pos: pos[:2] == 'NN'
        nouns = [word.lower() for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos) and word.isalpha() and len(word)>2]
        return list(set(nouns))
        
        
    ############ YT Video Metadata Extractor ##############
    def extract_meta_data(self, videoURL):        
        video = pafy.new(videoURL)
        tags = video.keywords
        title = self.noun_tokenizer(video.title)
        description = self.noun_tokenizer(video.description)
        return tags + title + description


    def get_relatedTags(self, relatedLinks):
        relatedTags = {k: [self.get_metaProperties(k)] for k in relatedLinks}
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
        self.model = model
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
        self.get_title = kwargs.get('get_title', True)
        self.get_description = kwargs.get('get_description', True)
        self.get_original_tags = kwargs.get('get_original_tags', True)
        self.rising = kwargs.get('rising_trends', True)   




    def setGranularity(self, granularity):
        self.granularity = granularity




    def setLanguage(self, language):
        self.catIds = utils.CategoryRead(geo=self.languages[language])




    def setDomain(self, domain):
        self.domain = domain




    def getCandidateTrends(self):
        self.candidate_trends = self.catIds.set_related_searches(self.domain, self.Gcategory, rising = self.rising)

    
    
    def singularize(self, tokens):
        wnl = WordNetLemmatizer()
        return [wnl.lemmatize(wrd) for wrd in tokens]
    
    

    def getTags(self, videoId):
        videoURL = f'https://www.youtube.com/watch?v={videoId}'
        yt_handler = YouTubeMetaExtractor(videoURL, get_original_tags=self.get_original_tags, get_title=self.get_title, get_description=self.get_description)
        #textual_meta = yt_handler.get_tags(videoURL)
        #textual_meta = yt_handler.properties
        textual_meta = self.singularize(extract_meta_data(videoURL))
        parsed = get_tags_sentence(textual_meta, word2vec=self.model)
        self.getCandidateTrends()
        if len(parsed) == 0:
            return []
        suggested_trends = self.selector.select_trends(parsed,
                          self.candidate_trends,
                          self.n_trends,
                          self.vectorizer)
        suggested_tags_from_metainfo = self.selector.select_trends([self.domain],
                          parsed,
                          self.n_trends,
                          self.vectorizer)
        return suggested_tags_from_metainfo, suggested_trends


