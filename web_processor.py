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


#*****************************************************************************
#***********************   Parameters settings    ****************************
#*****************************************************************************


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
opener = startOpener()
#opener.open('https://www.youtube.com')




#*****************************************************************************
#**********************   Functions definition   *****************************
#*****************************************************************************

def str2bool(string):
    if type(string) is bool: return string
    if (string == 'False') or (string == 'false'):
        return False
    return True



@app.route("/")
def main():
    global busy
    if busy:
        return 'System is busy...try again later...'
    return "Welcome! System is ready to start. Put video Id in the url address in the form: hosturl/api/id=videoid"




@app.route("/api/stop")
def stop():
    func = request.environ.get('werkzeug.server.shutdown')
    func()
    return "Server stopped!"




@app.route("/api/status")
def status():
    global busy
    if busy:
        return 'System is busy...try again later...'
    return "Server is running and waiting for next video. Put video Id in the url address in the form: hosturl/api/id=<videoid>"




@app.route(f'/api')
def process_video():
    global busy
    if busy:
        return 'System is busy...try again later...'
    busy = True
    videoid =request.args.get('id', default='', type = str)
    videoURL = 'https://www.youtube.com/watch?v=' + videoid
    resString = ''
    try:
        vdomain = request.args.get('domain', default=domain, type = str)
        gen_thumb = str2bool(request.args.get('gen_thumb', default=generate_thumbnails, type = str))
        gen_tags = str2bool(request.args.get('gen_tags', default=generate_tags, type = str))
        if gen_thumb:
            nframes = request.args.get('nframes', default=n_max_frames, type = int)
            cth = request.args.get('cth', default=corr_threshold, type = float)
            fth = request.args.get('fth', default=fsi_threshold, type = float)
            close_up_r = request.args.get('close_up_ratio', default=close_up_ratio, type = float)
            faces = str2bool(request.args.get('faces', default=process_faces, type = str))
            tmethod = request.args.get('method', default=method, type = str)
            smiles = str2bool(request.args.get('smiles', default=smile_detection, type = str))
            open_eyes = str2bool(request.args.get('open_eyes', default=open_eye_detection, type = str))
            m_length = request.args.get('max_length', default=max_length, type = int)
            thumb_parameters = {'domain': vdomain,
                    'log': LOG,
                    'n_max_frames': nframes,
                    'process_faces': faces,
                    'corr_threshold': cth,
                    'fsi_threshold': fth,
                    'smiles' : smiles,
                    'open_eyes': open_eyes,
                    'max_length': m_length,
                    'close_up_ratio': close_up_r}
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
                return 'ERROR: No valid method has been selected'
            thumb_handler.processVideo(videoURL, f'{output_folder_thumbnails}')
            resString += f'\r\rThumbnails generated and stored at folder: {output_folder_thumbnails}/{videoid}'
        if gen_tags:
            lang = request.args.get('lang', default=language, type = str)
            ntags = request.args.get('ntags', default=n_suggested_tags, type = int)
            gran =request.args.get('gran', default=granularity, type = str)
            title = str2bool(request.args.get('get_title', default=get_title, type = str))
            description = str2bool(request.args.get('get_description', default=get_description, type = str))
            original_tags = str2bool(request.args.get('get_original_tags', default=get_original_tags, type = str))
            rising = str2bool(request.args.get('rising_trends', default=rising_trends, type = str))
            tag_parameters  =  {'domain': vdomain,
                                'log': LOG,
                                'language': lang,
                                'n_suggested_tags': ntags,
                                'granularity': gran,
                                'get_title': title,
                                'get_description': description,
                                'get_original_tags': original_tags,
                                'rising_trends': rising,
                                'opener': opener}
            tag_handler = TagGenerator(model=models[lang], **tag_parameters)
    
            stm, st, stt, ch, tt, yt = tag_handler.getTags(videoid)
            resString += f'''\n\nGenerated tags:\n\n- Channel Name: {ch}\n\n- 
            Tags from title: {tt}\n\n- Tags from textual metadata: {stm}\n\n- 
            Trends from title: {stt}\n\n- Trends from category: {st}\n\n- 
            YouTube search bar suggestions: {yt}\n\n'''
            local_folder = f'{output_folder_tags}/{videoid}'
            if not os.path.exists(local_folder):
                os.makedirs(local_folder)
            with open(f'{local_folder}/GeneratedTags.txt', 'w') as f:
                f.write(resString)
        a = jsonify(success=True)
        busy = False
        return f'\r\rVideo {videoid} successfully processed.{resString}'
    except Exception as e:
        print(e)
        print(videoURL + " cannot be processed.")
        busy = False
        ts = time.gmtime()
        ts_readable = time.strftime("%Y-%m-%d %H:%M:%S", ts)
        with open(f'./ErrorsLog.txt', 'a+') as f:
            f.write(f'{ts_readable} - ERROR for video {videoid}:\n{e}')
        return 'ERROR: ' + str(e)



@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request. %s', e)
    ts = time.gmtime()
    ts_readable = time.strftime("%Y-%m-%d %H:%M:%S", ts)
    with open(f'./ErrorsLog.txt', 'a+') as f:
        f.write(f'{ts_readable} - An error occurred during a request for video {videoid}:\n{e}')
    return 'ERROR: ' + str(e)

@app.errorhandler(InternalServerError)
def handle_500(e):
    original = getattr(e, "original_exception", None)

    if original is None:
        # direct 500 error, such as abort(500)
        return render_template("500.html"), 500

    # wrapped unhandled error
    return render_template("500_unhandled.html", e=original), 500

###############  Server starting method ###########################

try:
    app.run(port = PORT_NUMBER)


except KeyboardInterrupt:
      print('^C received, shutting down the web server')
      func = request.environ.get('werkzeug.server.shutdown')
      func()

