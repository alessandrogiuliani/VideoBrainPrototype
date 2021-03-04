"""

Dependencies notes:
    Be sure that the following modules are installed and available in your
    Python environment:

    - PySceneDetect
    - OpenCV
    - Youyube-dl
    - Tensorflow
    - Pandas
    - Flask
    - flask_cors (in some releases of Flask, it should be missing)
"""
import logging
import os
import logging.config
from flask import Flask, jsonify, request
from fsi import FSI
from dod import DOD
from generic_thumbnail_processor import BFP, CFP
from automated_tag_enrichment import TagGenerator
from flask import request
from config import *
from werkzeug.exceptions import InternalServerError
import time

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

print(opener)



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
                    'max_length': m_length}
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
            tag_parameters  =  {'domain': vdomain,
                                'log': LOG,
                                'language': lang,
                                'n_suggested_tags': ntags,
                                'granularity': gran}
            tag_handler = TagGenerator(model=model, **tag_parameters)

            tags = tag_handler.getTags(videoid)
            tagString = '\n'.join(tags)
            resString += f'\r\rGenerated tags:\r\r{tagString}'
            local_folder = f'{output_folder_tags}/{videoid}'
            if not os.path.exists(local_folder):
                os.makedirs(local_folder)
            with open(f'{local_folder}/GeneratedTags.txt', 'w') as f:
                f.write(tagString)
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

