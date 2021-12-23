# -*- coding: utf-8 -*-
"""
@credits: Leonardo Piano, Alessandro Giuliani
@author: Leonardo Piano, Alessandro Giuliani

This module contains all functionalities for the Fast Scene Identification
(FSI) algorithm.
"""

import glob
import ntpath
import logging
import os
import cv2
import time
import logging.config
import my_pafy as pafy
import pandas as pd
from my_scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
import shutil
from yolo import YOLO
from PIL import Image
from six.moves.urllib.parse import urlparse
from scenedetect.frame_timecode import FrameTimecode


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
yoloInstance = YOLO()



class FSI(object):

    logFormatter = logging.Formatter("""%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s""")
    logger = logging.getLogger()
    

    def __init__(self, **kwargs):
        self.domain = kwargs.get('domain', None)
        self.n_max_frames = kwargs.get('n_max_frames', 5)
        self.log = kwargs.get('log', False)
        self.fsi_threshold = kwargs.get('fsi_threshold', 15)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        self.eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smiles = kwargs.get('smiles', True) 
        self.open_eyes = kwargs.get('open_eyes', True)
        self.max_length = kwargs.get('max_length', 0)     
        self.process_faces = kwargs.get('process_faces', False)
        self.close_up_ratio = kwargs.get('close_up_ratio', 0.1)
        self.opener = kwargs.get('opener', None)

    def cleanDir(self, folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    
    
    
    
    def is_url(self, path):
        try:
            result = urlparse(path)
            return result.scheme and result.netloc and result.path
        except Exception:
            return False
    
    
    
    
    def getVideoId(self, url):
        videoid = ''
        if self.is_url(url):
            l = url.split('?v=')
            videoid = l[len(l)-1]
        return videoid



    
    def getBestVideo(self, video, width_limit=10000):
        if self.log: print("-- Getting best video")
        streams = video.allstreams
        res = None
        for s in streams:
            if s.mediatype in ('normal', 'video') and \
                    s.extension=='mp4' and \
                    ('av01' not in s._info.get('vcodec')):  
                width = s.dimensions[0]
                if width <= width_limit:
                    if res is None:
                        res = s
                        continue
                    if  width > res.dimensions[0]: res = s
        return res.url
    
    
    def getNormalVideo(self, video, width_limit=10000):
        if self.log: print("-- Getting best video")
        streams = video.allstreams
        res = list()
        for s in streams:
            if s.mediatype == 'normal' and \
                    ('av01' not in s._info.get('vcodec')):  
                res.append(s.url)
        return res
    
    
    def path_leaf(self, path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)
    
    
    
    
    def createFolder(self, videoid):
        newpath = os.getcwd()+'/outputs1/' + videoid
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        return newpath
    
    
    
    
    def createWorkDir(self, video_id, folder):
        newpath = folder + '/' + video_id
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        else:
            self.cleanDir(newpath)
        return newpath
    
    
    
    
    def predict_img(self, frame):
        prediction = []
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        prediction = yoloInstance.detect_img(image, self.domain)
        return prediction
    
    
    
    
    def differentsFrames(self, video):
        if video is None: return
        video_manager = VideoManager([video])
        if self.max_length > 0:
            video_manager.set_duration(start_time = video_manager.get_base_timecode(),
                                       duration = FrameTimecode(timecode=float(self.max_length), 
                                                                fps = video_manager.get_framerate()))
        scene_manager = SceneManager(StatsManager())
        scene_manager.add_detector(ContentDetector(self.fsi_threshold, min_scene_len=15))
        base_timecode = video_manager.get_base_timecode()
        try:
            # Set downscale factor to improve processing speed.
            video_manager.set_downscale_factor()
            # Start video_manager.
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            # Obtain list of detected scenes.
            scene_list = scene_manager.get_scene_list(base_timecode)
            frames = []
            for i, scene in enumerate(scene_list):
                frames.append(scene[0].get_frames())
        except Exception as e:
            print(f'An error occurred in scene detection. Reason: {e}')
        finally:
            video_manager.release()
        return frames
    
    
    
    
    def getOnlyNFramesFaces(self, best_frames):
        while len(best_frames) > self.n_max_frames:
            predictions = best_frames[0]['predictions']
            max_area, frame_id, counter = 0, 0, 0
            for frame in best_frames:
                predictions = frame['predictions']
                for prediction in predictions:
                    if prediction['area'] > max_area:
                        max_area = prediction['area']
                        frame_id = counter
                counter += 1
            del best_frames[frame_id]
        return best_frames
    
    
    
            
    def getOnlyNFramesYolo(self, best_frames):
        while len(best_frames) > self.n_max_frames:
            predictions = best_frames[0]['predictions']
            min = float(predictions[0]['score'])
            frame_id, counter = 0, 0
            for frame in best_frames:
                predictions = frame['predictions']
                for prediction in predictions:
                    if float(prediction['score']) < min:
                       frame_id = counter
                       min = prediction['score']
                counter += 1
            del best_frames[frame_id]
        return best_frames
    
    
    
    
            
    def getFinalThumbnails(self, workdir, metadata):
        bestFrames = []
        max_score = 0
        print("FRAMES  ", len(metadata))
        if len(metadata) >= self.n_max_frames:
            for frame in metadata:
                predictions = frame['predictions']
                best = None
                for prediction in predictions:
                    if float(prediction['score']) > max_score:
                        best = frame
                if best is not None:
                    bestFrames.append(best)
            if(len(bestFrames) > self.n_max_frames and self.domain != 'music'):
                bestFrames = self.getOnlyNFramesYolo(bestFrames)
            else:
                bestFrames = self.getOnlyNFramesFaces(bestFrames)
            for file in bestFrames:
                fnumb = file['fnumb']
                fname = file['fname']
                old = workdir+'/'+fname
                new_name = workdir+ '/'+'finalThumb_'+str(fnumb)+'.jpg'
                os.rename(old, new_name)
            images = glob.glob(workdir+'/'+'*')
            for filename in images:
                if 'localMaxFrame' in filename:
                    os.unlink(filename)
        else:
            images = glob.glob(workdir+'/'+'*')
            for filename in images:
                os.rename(filename, filename.replace('localMaxFrame', 'finalThumb'))

    
    
        
        
    def getImgseries(self, video_url, outputFolder):
        prediction, metadata, framesList = [], [], []
        video = video_url
        if self.log: print("video_url "+str(video_url))
        #it returns the id if is the url of a youtube video
        v_id = self.getVideoId(video)
        if self.log: print("v_id "+str(v_id))
        if self.is_url(video):
            if self.log: print("before pafy")
            if self.opener is not None: self.opener.open(video)
            videoPafy = pafy.new(video)
            if self.opener is not None: self.opener.close()
            if self.log: print("after pafy")
            video = self.getBestVideo(videoPafy)
        else:
            v_id = ''  
        workdir = self.createWorkDir(v_id, outputFolder)
        try:
            framesList = self.differentsFrames(video)
        except:
            if self.opener is not None: self.opener.open(video_url)
            videoPafy = pafy.new(video_url)
            video = self.getNormalVideo(videoPafy)[0]
            framesList = self.differentsFrames(video)
            if self.opener is not None: self.opener.close()
        print("Detected Frames : ", len(framesList))
        cam = cv2.VideoCapture(video)
        if not cam.isOpened():
            raise IOError('Can\'t open Yolo2Model')
        frame_num = 0
        fps = cam.get(cv2.CAP_PROP_FPS)
        frame_limit = fps * self.max_length
        try:
            while True:
                ret, frame = cam.read()
                if not ret:
                    if self.log: print('Can\'t read video data. Potential end of stream')
                    return self.getFinalThumbnails(workdir, metadata)
                if frame_num in framesList:
                    if (self.domain == 'music'):
                        prediction = self.predictFaces(frame)
                    else:
                        prediction = self.predict_img(frame)
                        if (self.process_faces is True):
                            prediction += self.predictFaces(frame)
                    if len(prediction) > 0:
                        meta_info = {}
                        fname = 'localMaxFrame_'+str(frame_num) + '.jpg'
                        cv2.imwrite(f'{outputFolder}/{v_id}/localMaxFrame_{frame_num}.jpg', frame)
                        meta_info['fname'] = fname
                        meta_info['predictions'] = prediction
                        meta_info['fnumb'] = frame_num
                        metadata.append(meta_info)
                    if len(prediction) == 0 and len(framesList) <= self.n_max_frames:
                        meta_info = {}
                        fname = f'localMaxFrame_{str(frame_num)}.jpg'
                        cv2.imwrite(f'{outputFolder}/{v_id}/localMaxFrame_{frame_num}.jpg', frame)
                        meta_info['fname'] = fname
                        meta_info['predictions'] = prediction
                        meta_info['fnumb'] = frame_num
                        metadata.append(meta_info)
                time.sleep(0.000001)           
                key = cv2.waitKey(1) & 0xFF
                # Exit
                if key == ord('q'):
                    break
                frame_num += 1
                if self.max_length > 0:
                    if frame_num > frame_limit:
                        cam.release()
                        return self.getFinalThumbnails(workdir, metadata)
                        break
        finally:
            cam.release()

    
    

    def predictFaces(self, image):
        # Load the cascade
        # face_cascade = cv2.CascadeClassifier(f'{os.getcwd()}/model_data/haarcascade_frontalface_default.xml')
        # Convert into grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        window = (int(gray.shape[0]*self.close_up_ratio), int(gray.shape[1]*self.close_up_ratio))
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=window)
        predictions = []
        # add the faces
        for (x, y, w, h) in faces:
            if self.smiles:
                roi_gray = gray[y:y + h, x:x + w]
                sm = self.smile_cascade.detectMultiScale(roi_gray, 1.1, 4)
                if len(sm) == 0: continue
            if self.open_eyes:
                roi_gray = gray[y:y + h, x:x + w]
                eyes = self.eyes_cascade.detectMultiScale(roi_gray, 1.1, 4)
                if len(eyes) == 0: continue
            prediction = {"area": int(w*h),
                         "box": [int(x), int(y), int(w), int(h)],
                         "class": "face",
                         "score": 0.99}
            predictions.append(prediction)
        return predictions
    
    
    
    
    def processVideo(self, videoURL, outputFolder):
        if self.log:
            print("Processing at: "+os.getcwd())
        self.opener.open(videoURL)
        self.getImgseries(videoURL, outputFolder)
        self.opener.close()
        print("-- Extract frame ok")
    


    
    def processCSV(self, outputFolder, csvfile):
        data = pd.read_csv(csvfile)
        toProcess = ['https://www.youtube.com/watch?v='+l for l in data['itemCode']]
        for url in toProcess:
            try:
                print('Processing video ' + url)
                checkFolder = f'{outputFolder}{self.getVideoId(url)}/'
                if os.path.isdir(checkFolder):
                    print('Checked, maybe empty dir')
                    if len(os.listdir(checkFolder)) > 0:
                        print("Video already processed")
                        continue
                start = time.time()
                self.processVideo(url, outputFolder)
                print('Processing video completed!!')
                print('Time: ',( time.time() - start))
            except:
                logging.exception('ERROR')
                print(url + " cannot be processed.")
            
