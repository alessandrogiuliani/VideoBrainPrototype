"""
@author: Alessandro Giuliani

"""
import logging
import os
import logging.config
from flask import Flask, jsonify, request, render_template
from fsi import FSI
from dod import DOD
from generic_thumbnail_processor import BFP, CFP
from automated_tag_enrichment import TagGenerator
from flask import request
from config import *
from werkzeug.exceptions import InternalServerError
import time
from urllib.error import HTTPError
import argparse
from gensim.models import Word2Vec,KeyedVectors
from gensim.models.wrappers import FastText
import os
import googleapiclient.discovery
import googleapiclient.errors
from oauth2client import client # Added
from oauth2client import tools # Added
from oauth2client.file import Storage # Added
import pafy


############ Sessions and authentication ###############
# def connect_api(client_secrets_file, credential_sample_file):
#     store = Storage(os.path.join('./', credential_sample_file))
#     credentials = store.get()
#     if not credentials or credentials.invalid:
#         flow = client.flow_from_clientsecrets(client_secrets_file, scopes)
#         credentials = tools.run_flow(flow, store)
#     youtube = googleapiclient.discovery.build(
#         api_service_name, api_version, credentials=credentials)
#     return youtube

# scopes = ["https://www.googleapis.com/auth/youtube.readonly"]
# api_service_name = "youtube"
# api_version = "v3"
# client_secrets_file = "secret.json"
# credential_sample_file = 'credential_sample.json'
# youtube = connect_api(client_secrets_file, credential_sample_file)
API_KEY = 'AIzaSyDC3IWQ-Ugkzn_aKsi3NdMFkyUsj6_KKW0'
pafy.set_api_key(API_KEY)

#*****************************************************************************
#***********************   Parameters settings    ****************************
#*****************************************************************************

def str2bool(string):
    if (string == 'False') or (string == 'false'):
        return False
    return True

#*****************************************************************************
#*************************   Inizialization   ********************************
#*****************************************************************************

#Internal variables
busy = False
app = Flask(__name__, static_url_path= STATIC_URL_PATH)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('flask_cors').level = logging.DEBUG
#CORS(app, resources=r'/api/*')
#opener = startOpener()
#opener.open('https://www.youtube.com')

parser = argparse.ArgumentParser(description='VideoBrain prototype started!')



#Framework parameters
parser.add_argument('-PORT', '--PORT_NUMBER', required=False, default=PORT_NUMBER, type=int)
parser.add_argument('-LOG', '--LOG', required=False, default=LOG, type=str2bool)
parser.add_argument('-E', '--EMBEDDING', required=False, default=False, type=str2bool)

#Main parameters
parser.add_argument('-d', '--domain', required=False, default=domain, type=str)
parser.add_argument('-t', '--gen_thumb', required=False, default=generate_thumbnails, type=str2bool) 
parser.add_argument('-T', '--gen_tags', required=False, default=generate_tags, type=str2bool) 
parser.add_argument('-i', '--id',required=True, type=str)

#Thumbnail generator parameters
parser.add_argument('-o', '--output_folder_thumbnails', required=False, default=output_folder_thumbnails, type=str)
parser.add_argument('-n', '--nframes', required=False, default=n_max_frames, type=int)
parser.add_argument('-m', '--method', required=False, default=method, type=str)
parser.add_argument('-cth', '--cth', required=False, default=corr_threshold, type=float)
parser.add_argument('-fth', '--fth', required=False, default=fsi_threshold, type=float)
parser.add_argument('-f', '--faces', required=False, default=process_faces, type=str2bool)
parser.add_argument('-s', '--smiles', required=False, default=smile_detection, type=str2bool)
parser.add_argument('-e', '--open_eyes', required=False, default=open_eye_detection, type=str2bool)
parser.add_argument('-l', '--max_length', required=False, default=max_length, type=int)
parser.add_argument('-c', '--close_up_ratio', required=False, default=close_up_ratio, type=float)

#Tag generator parameters
parser.add_argument('-N', '--ntags', required=False, default=n_suggested_tags, type=int)
parser.add_argument('-L', '--lang', required=False, default=language, type=str)
parser.add_argument('-G', '--gran', required=False, default=granularity, type=str)
parser.add_argument('-O', '--output_folder_tags', required=False, default=output_folder_tags, type=str)
parser.add_argument('-GT', '--get_title', required=False, default=get_title, type=str2bool)
parser.add_argument('-GD', '--get_description', required=False, default=get_description, type=str2bool)
parser.add_argument('-GO', '--get_original_tags', required=False, default=get_original_tags, type=str2bool)
parser.add_argument('-TT', '--top_trends', required=False, default=top_trends, type=str2bool)
parser.add_argument('-R', '--rising_trends', required=False, default=rising_trends, type=str2bool)


args = parser.parse_args()
resString = ''
videoURL = 'https://www.youtube.com/watch?v=' + args.id
if args.EMBEDDING:
    models = dict()
    models['italian']= FastText.load_fasttext_format(f'{os.getcwd()}/model_data/it')
    vec = f'{os.getcwd()}/model_data/GoogleNews-vectors-negative300.bin.gz'
    models['english'] = KeyedVectors.load_word2vec_format(vec, binary=True)
    
    #models['english']= Word2Vec(abc.sents())   #only for testing
    for key, value in models.items():
        value.init_sims(replace=True)
if args.gen_thumb:
    thumb_parameters = {'domain': args.domain,
                'log': args.LOG,
                'n_max_frames': args.nframes,
                'process_faces': args.faces,
                'corr_threshold': args.cth,
                'fsi_threshold': args.fth,
                'smiles': args.smiles,
                'open_eyes' : args.open_eyes,
                'max_length': args.max_length,
                'close_up_ratio': args.close_up_ratio}
    tmethod = args.method
    if tmethod == 'FSI': 
        thumb_handler = FSI(**thumb_parameters)
    elif tmethod == 'DOD':
        thumb_handler = DOD(**thumb_parameters)
    elif tmethod == 'BFP':
        thumb_handler = BFP(**thumb_parameters)
    elif tmethod == 'CFP':
        thumb_handler = CFP(**thumb_parameters)
    else:
        print('ERROR: No valid method has been selected')
        exit(1)
    thumb_handler.processVideo(videoURL, f'{args.output_folder_thumbnails}')
    resString += f'\r\rThumbnail generated and stored at folder: {args.output_folder_thumbnails}'
if args.gen_tags:
    tag_parameters  =  {'domain': args.domain,
                        'log': args.LOG,
                        'language': args.lang,
                        'n_suggested_tags': args.ntags,
                        'granularity': args.gran,
                        'get_title': args.get_title,
                        'get_description': args.get_description,
                        'top_trends': args.top_trends,
                        'rising_trends': args.rising_trends}
    tag_handler = TagGenerator(model=models[args.lang], **tag_parameters)
    suggested_tags_from_metainfo, suggested_trends = tag_handler.getTags(args.id)
    resString += f'\n\nGenerated tags:\nFrom video metadata: {suggested_tags_from_metainfo}\nFrom trends: {suggested_trends}\n\n'
#a = jsonify(success=True)
print(resString)

