# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:20:55 2020

@credits: Leonardo Piano
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
import pafy
import pandas as pd
from my_scenedetect.video_manager import VideoManager
from my_scenedetect.scene_manager import SceneManager
from my_scenedetect.stats_manager import StatsManager
from my_scenedetect.detectors import ContentDetector
import shutil
from yolo import YOLO
from PIL import Image
from six.moves.urllib.parse import urlparse



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



    
    def getBestVideo(self, video):
        if self.log: print("-- Getting best video")
        streams = video.streams
        self.logger.info(streams)
        return video.getbest(preftype="mp4").url
    
    
    
    
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
        for item in prediction:
            if item['class'] == 'person':
                item['score'] = item['score'] * 0.01
        return prediction
    
    
    
    
    def differentsFrames(self, video):
        if video is None: return
        video_manager = VideoManager([video])
        scene_manager = SceneManager(StatsManager())
        scene_manager.add_detector(ContentDetector(self.fsi_threshold))
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
                new_name = workdir+ '/'+'finalThumb '+str(fnumb)+'.jpg'
                os.rename(old, new_name)
            images = glob.glob(workdir+'/'+'*')
            for filename in images:
                if 'localMaxFrame' in filename:
                    os.unlink(filename)
    
    
        
        
    def getImgseries(self, video_url, outputFolder):
        prediction, metadata, framesList = [], [], []
        start_time = time.time()
        video = video_url
        if self.log: print("video_url "+str(video_url))
        #it returns the id if is the url of a youtube video
        v_id = self.getVideoId(video)
        if self.log: print("v_id "+str(v_id))
        if self.is_url(video):
            if self.log: print("before pafy")
            videoPafy = pafy.new(video)
            if self.log: print("after pafy")
            video = self.getBestVideo(videoPafy)
        else:
            v_id = ''
        workdir = self.createWorkDir(v_id, outputFolder)
        framesList = self.differentsFrames(video)
        print("Detected Frames : ", len(framesList))
        cam = cv2.VideoCapture(video)
        if not cam.isOpened():
            raise IOError('Can\'t open Yolo2Model')
        frame_num = 0
        fps = 0
        try:
            while True:
                ret, frame = cam.read()
                if not ret:
                    if self.log: print('Can\'t read video data. Potential end of stream')
                    return self.getFinalThumbnails(workdir, metadata)
                if frame_num in framesList:
                    if self.domain == 'music':
                        prediction = self.predictFaces(frame)
                    else:
                        prediction = self.predict_img(frame)
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
                end_time = time.time()
                fps = fps * 0.9 + 1/(end_time - start_time) * 0.1
                start_time = end_time
                time.sleep(0.000001)           
                key = cv2.waitKey(1) & 0xFF
                # Exit
                if key == ord('q'):
                    break
                frame_num += 1
        finally:
            cam.release()

    
    

    def predictFaces(self, image):
        # Load the cascade
        face_cascade = cv2.CascadeClassifier(f'{os.getcwd()}/haarcascade_frontalface_default.xml')
        # Convert into grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        predictions = []
        # add the faces
        for (x, y, w, h) in faces:
            prediction = {"area": int(w*h),
                         "box": [int(x), int(y), int(w), int(h)],
                         "class": "face",
                         "score": 0.99}
            predictions.append(prediction)
        return predictions
    
    
    
    
    def processVideo(self, videoURL, outputFolder):
        if self.log:
            print("Processing at: "+os.getcwd())
        self.getImgseries(videoURL, outputFolder)
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
            