# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:42:30 2020

@author: Alessandro Giuliani, Maria Madalina Stanciu

"""
from pandas.io.json import json_normalize
import json
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
import my_pafy as pafy
from nltk.stem import WordNetLemmatizer
from config import startOpener


#*****************************************************************************
#***********************  Global initialization   ****************************
#*****************************************************************************
warnings.filterwarnings('ignore')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('abc')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

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
        if stop_words == 'italian':
            with open(f'{os.getcwd()}/model_data/stopwords-it.txt') as f:
                self.stop_words = [line.replace('\n', '') for line in f.readlines()]
                return
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
        self.opener = kwargs.get('opener', None)
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

        
        
       

    def get_channel(self, link):
        if self.opener is not None: self.opener.open(link)
        webpage = urlopen(link).read()
        soup = BeautifulSoup(webpage, features="lxml", from_encoding="iso-8859-1")
        if self.opener is not None: self.opener.close()
        return soup.find_all("link",  itemprop="name")[0]['content']



    def get_relatedSearchLinks(self):
        queryParameter = '+'.join(self.properties['title'].split()).encode('utf-8')
        relatedSearch = f'{self.searchURLToken}{queryParameter}'
        print(relatedSearch)
        if self.opener is not None: self.opener.open(relatedSearch)
        temp = urlopen(relatedSearch)
        webpage = temp.read()
        soup = BeautifulSoup(webpage, 'lxml')
        data = soup.find_all("a", class_ = "yt-uix-tile-link yt-ui-ellipsis yt-ui-ellipsis-2 yt-uix-sessionlink spf-link")
        relatedUrls= [(self.videoURLToken + x.get('href')) for x in data if x.get('href') not in relatedSearch]
        if self.opener is not None: self.opener.close()
        return relatedUrls
        


    def get_tags(self, link):
        if self.opener is not None: self.opener.open(link)
        webpage = urlopen(link).read()
        soup = BeautifulSoup(webpage, "lxml")
        try:
            tags = soup.find_all("meta",  property="og:video:tag")
        except:
            if self.opener is not None: self.opener.close()
            return []
        if self.opener is not None: self.opener.close()
        return [x['content'] for x in tags]
    
    
    def retrieve_description(self, link):
        if self.opener is not None: self.opener.open(link)
        webpage = urlopen(link).read()
        soup = BeautifulSoup(webpage, features="lxml", from_encoding="iso-8859-1")
        description = soup.find_all("meta",  property="og:description")[0]['content']
        if self.opener is not None: self.opener.close()
        return description
    
    

    def noun_tokenizer(self, string):
        tokenized = nltk.word_tokenize(string)
        is_noun = lambda pos: pos[:2] == 'NN'
        nouns = [word.lower() for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos) and word.isalpha() and len(word)>2]
        return list(set(nouns))
        
        
    ############ YT Video Metadata Extractor ##############
    def extract_meta_data(self, videoURL):
        if self.opener is not None: self.opener.open(videoURL)        
        video = pafy.new(videoURL)
        res = list()
        if self.get_original_tags:
            res += self.get_tags(videoURL)
        if self.get_title:
            res += self.noun_tokenizer(video.title)
        if self.get_description:
            res += self.noun_tokenizer(re.sub(r'(https)|(http)?:\/\/\S*', '', self.retrieve_description(videoURL), flags=re.MULTILINE)) 
        if self.opener is not None: self.opener.close()
        return res


    
    def get_title_tokens(self, videoURL):
        if self.opener is not None: self.opener.open(videoURL)
        video = pafy.new(videoURL)
        self.video_title = video.title
        if self.opener is not None: self.opener.close()
        return self.noun_tokenizer(self.video_title)
    
    
    
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
    languages = {'english': 'GB',
                 'italian': 'IT'}
    mapping_category = {'food': catIds.get_category_id('Food & Drink'),
                        'cars': catIds.get_category_id('Autos & Vehicles'),
                        'animals': catIds.get_category_id('Pets & Animals'),
                        'tech': catIds.get_category_id('Computers & Electronics'),
                        'music': catIds.get_category_id('Music & Audio'),
                        'sport': catIds.get_category_id('Sports'),
                        'news': catIds.get_category_id('News')}

    mapping_italian = {'food': ['Cibo', 'bevande'],
                        'cars': ['auto', 'moto'],
                        'animals': ['animali'],
                        'tech': ['tecnologia'],
                        'music': ['musica'],
                        'sport': ['sport'],
                        'news': ['notizie']} 


    def __init__(self, model=None, **kwargs):
        self.language = kwargs.get('language', 'english')
        self.model = model
        self.vectorizer= MySentences(self.model, stop_words=self.language)
        self.domain = kwargs.get('domain', None)
        if self.domain not in self.mapping_category.keys():
            print('Error: no valid domain selected')
        self.Gcategory = self.mapping_category[self.domain]
        granularity = kwargs.get('granularity', 'WL')
        self.selector = self.selectors[granularity]
        self.n_trends = kwargs.get('n_suggested_tags', 5)
        self.google_trends = {}
        self.catIds = utils.CategoryRead(geo=self.languages[self.language])
        self.get_title = kwargs.get('get_title', True)
        self.get_description = kwargs.get('get_description', True)
        self.get_original_tags = kwargs.get('get_original_tags', True)
        self.top = kwargs.get('top_trends', True) 
        self.rising = kwargs.get('rising_trends', True)   
        self.opener = kwargs.get('opener', None)   



    def setGranularity(self, granularity):
        self.granularity = granularity




    def setLanguage(self, language):
        self.catIds = utils.CategoryRead(geo=self.languages[language])




    def setDomain(self, domain):
        self.domain = domain




    def getCandidateTrends(self):
        self.candidate_trends = self.catIds.set_related_searches(self.domain, self.Gcategory, top=self.top, rising = self.rising)

    
    
    def getTitleTrends(self, title_tokens):
        results = list()
        for token in title_tokens:
            results += self.catIds.get_trends_by_keyword(token, top=self.top, rising = self.rising)
        self.title_trends = set(results)

        
        
    
    def singularize(self, tokens):
        wnl = WordNetLemmatizer()
        return [wnl.lemmatize(wrd) for wrd in tokens]
    
    
    
    def get_YT_suggestions(self, query):
        query_formatted = query.replace(' ', '%20')
        url = f'http://suggestqueries.google.com/complete/search?client=firefox&ds=yt&q={query_formatted}%20'
        if self.opener is not None: self.opener.open(url)
        response = urlopen(url)
        data_json = json.loads(response.read())
        if self.opener is not None: self.opener.close()
        return data_json[1]


    def getTags(self, videoId):
        videoURL = f'https://www.youtube.com/watch?v={videoId}'
        yt_handler = YouTubeMetaExtractor(videoURL, 
                                          get_original_tags=self.get_original_tags, 
                                          get_title=self.get_title, 
                                          get_description=self.get_description, 
                                          opener=self.opener)
        #textual_meta = yt_handler.get_tags(videoURL)
        #textual_meta = yt_handler.properties
        textual_meta = yt_handler.extract_meta_data(videoURL)
        #parsed = get_tags_sentence(textual_meta, word2vec=self.model)
        parsed = [word for word in textual_meta if word not in self.vectorizer.stop_words]
        self.getCandidateTrends()
        title_tokens = [word for word in yt_handler.get_title_tokens(videoURL) if word not in self.vectorizer.stop_words]
        self.getTitleTrends(title_tokens)
        channel_name = yt_handler.get_channel(videoURL)
        yt_suggestions = self.get_YT_suggestions(channel_name)[:self.n_trends]
        if len(parsed) == 0:
            return []
        if self.language == 'english': 
            domain = [self.domain]
        elif self.language == 'italian': 
            domain = self.mapping_italian[self.domain]
        try:
            suggested_trends = self.selector.select_trends(parsed,
                          self.candidate_trends,
                          self.n_trends,
                          self.vectorizer)
        except AttributeError:
            suggested_trends = []
        suggested_tags_from_metainfo = self.selector.select_trends(domain,
                          parsed,
                          self.n_trends,
                          self.vectorizer)
        suggested_trends_from_title = self.selector.select_trends(parsed,
                          self.title_trends,
                          self.n_trends,
                          self.vectorizer)
        return suggested_tags_from_metainfo, suggested_trends, suggested_trends_from_title, channel_name, title_tokens, yt_suggestions


