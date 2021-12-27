# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:42:30 2020

@author: Alessandro Giuliani

"""
import os
from configparser import ConfigParser
from gensim.models import Word2Vec,KeyedVectors
from gensim.models.wrappers import FastText
from nltk.corpus import abc
import sys
import requests





def str2bool(string):
    if (string == 'False') or (string == 'false'):
        return False
    return True


cfg = f'{os.getcwd()}/config/config.ini'


parser = ConfigParser()    
parser.read(cfg)


#Framework parameters
PORT_NUMBER = int(parser['framework']['PORT_NUMBER'])
API_KEY = parser['framework']['API_KEY']
STATIC_URL_PATH = parser['framework']['STATIC_URL_PATH']
LOG = str2bool(parser['framework']['LOG'])
load_embedding_model = str2bool(parser['framework']['load_embedding_model'])
luminati_username = parser['framework']['luminati_username']
luminati_password = parser['framework']['luminati_password']


#Main parameters
domain = parser['main']['domain']
generate_thumbnails = str2bool(parser['main']['generate_thumbnails'])
generate_tags = str2bool(parser['main']['generate_tags'])

#Thumbnail generator parameters
output_folder_thumbnails = parser['thumbnails']['output_folder_thumbnails']
n_max_frames = int(parser['thumbnails']['n_max_frames'])
method = parser['thumbnails']['method']
corr_threshold = float(parser['thumbnails']['corr_threshold'])
fsi_threshold = float(parser['thumbnails']['fsi_threshold'])
process_faces = str2bool(parser['thumbnails']['process_faces'])
smile_detection = str2bool(parser['thumbnails']['smile_detection'])
open_eye_detection = str2bool(parser['thumbnails']['open_eye_detection'])
max_length = int(parser['thumbnails']['max_length'])
close_up_ratio = float(parser['thumbnails']['close_up_ratio'])

#Tag generator parameters
n_suggested_tags = int(parser['tags']['n_suggested_tags'])
language = parser['tags']['language']
granularity = parser['tags']['granularity']
output_folder_tags = parser['tags']['output_folder_tags']
get_title = str2bool(parser['tags']['get_title'])
get_description = str2bool(parser['tags']['get_description'])
get_original_tags = str2bool(parser['tags']['get_original_tags'])
rising_trends = str2bool(parser['tags']['rising_trends'])
top_trends = str2bool(parser['tags']['top_trends'])


proxy = f'http://{luminati_username}:{luminati_password}@zproxy.lum-superproxy.io:22225'

def startOpener()  :
    if sys.version_info[0]==2:
        import six
        from six.moves.urllib import request
        opener = request.build_opener(
            request.ProxyHandler(
                {f'http': proxy,
                f'https': proxy}))
    if sys.version_info[0]==3:
        import urllib.request
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler(
                {'http': proxy,
                'https': proxy}))
    return opener



