# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:20:55 2020

@credits: Leonardo Piano
@author: Leonardo Piano, Alessandro Giuliani


This module contains all functionalities for the Dynamic Object Detection (DOD) algorithm.

"""

from yolo import YOLO
from PIL import Image
from six.moves.urllib.parse import urlparse
import glob
import ntpath
import logging
import os
import cv2
import time
import logging.config
import pafy
import numpy as np
import json
import pandas as pd
import time
import my_progress_bar

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil

widgets=[
    ' [', my_progress_bar.Percentage(), '] ',
    my_progress_bar.Bar(marker='█'),
    ' (', my_progress_bar.ETA(), ') ',
]
yoloInstance = YOLO()

class DOD(object):

    logFormatter = logging.Formatter("""%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s""")
    logger = logging.getLogger()




    def __init__(self, **kwargs):
        self.domain = kwargs.get('domain', None)
        self.n_max_frames = kwargs.get('n_max_frames', 5)
        self.log = kwargs.get('log', False)
        self.corr_threshold = kwargs.get('corr_threshold', 0.9)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        self.eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smiles = kwargs.get('smiles', True) 
        self.open_eyes = kwargs.get('open_eyes', True)
        self.max_length = kwargs.get('max_length', 0)    
            
        


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
        return video.getbest(preftype="mp4").url
    
    
    
    
    def path_leaf(self, path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)
    
    
    
    
    def createWorkDir(self, video_id, folder):
        newpath = folder + '/' + video_id
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        else:
            self.cleanDir(newpath)
        return newpath
    



    def estimate_blur(self, image):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_map = cv2.Laplacian(image, cv2.CV_64F)
        score = np.var(blur_map)
        return score




    def predict_img(self, img, compair):
        prediction=[]
        dist = self.compareHist(img, compair, isFile = False) if compair is not None else 0       
        if dist <= self.corr_threshold:
            if self.domain == 'music':
                prediction = self.predictFaces(img)
            else:
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                prediction = yoloInstance.detect_img(image, self.domain)
            for item in prediction:
                if item['class'] == 'person':
                    item['score'] = item['score'] * 0.01
        return prediction

        

    
    def compareHist(self, file1, file2, stat=cv2.HISTCMP_CORREL, isFile = True):
        target_im = cv2.imread(file1) if isFile else file1
        target_hist = cv2.calcHist([target_im], [0], None, [256], [0, 256])
        comparing_im = cv2.imread(file2) if isFile else file2
        comparing_hist = cv2.calcHist([comparing_im], [0], None, [256], [0, 256])
        diff = cv2.compareHist(target_hist, comparing_hist, stat)
        return diff 
    
    
    
    
    def batchHistCompare(self, videofolder, predictions, corr=0.99):
        metadata = []
        if corr > 1:
            return metadata
        try:
            #self.logger.info(videofolder)
            video_id = self.path_leaf(videofolder)
            files = glob.glob(f'{videofolder}/*')
            files.sort(key=os.path.getmtime)
            count, scene_counter = 0, 0
            target = files[0]
            for file in files:
                try:
                    fname = self.path_leaf(file)
                    #if self.log: print(file)
                    dist = self.compareHist(target, file, isFile=True)
                    target = file
                    isChanged = " no scene change."
                    if (dist < corr): 
                        isChanged = " THIS IS A NEW SCENE."
                        scene_counter += 1 
                        
                    #if self.log: print(f'Image {count}, {self.path_leaf(file)} compare {dist}. {isChanged}')
                    metainfo = {'file': file,
                                'frame': fname.split('_')[1],
                                'blur': fname.split('_')[2].split('.')[0],
                                'scene': scene_counter,
                                'corr': dist,                                                                
                                'predictions': predictions[count]}
                    metadata.append(metainfo)
                    count += 1
                except:
                    self.logger.error(f'video id {video_id} not processed image {file}.', exc_info=True)
        except Exception as e:
            self.logger.error(f'video id {video_id} not processed.')
            #print(e)
        return metadata

    
    
    
    def getImgseries(self, video_url, v=''):
        metadata, target, series = list(), list(), list()
        if self.log: print(f'Downloading video: {video_url}')
        v_id = self.getVideoId(video_url)    
        if self.is_url(video_url):
            videoPafy = pafy.new(video_url)  
            self.bestVideo = self.getBestVideo(videoPafy)
        cam = cv2.VideoCapture(self.bestVideo)
        if not cam.isOpened():
            raise IOError('Can\'t open Yolo2Model')
        frame_num = 0
        fps = cam.get(cv2.CAP_PROP_FPS)
        frame_limit = fps * self.max_length
        ret, target = cam.read()
        while True:
            ret, frame = cam.read()
            if not ret:
                if self.log: print('End of stream')
                self.totalFrames = frame_num +1
                cam.release()
                print('Frames extracted:     ' + str(frame_num))
                return series, metadata
            frame_num += 1
            prediction = self.predict_img(frame, target)
            target = frame
            if len(prediction) > 0:
                series.append(frame_num)
                metadata.append(prediction)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if self.max_length > 0:
                if frame_num > frame_limit:
                    cam.release()
                    print('Frames extracted:     ' + str(frame_num))
                    self.totalFrames = frame_num +1
                    return series, metadata


    
    
    
    def extractFrames(self, video_url, frame_series, outputFolder, v=''):
        if self.log: print("Extracting frames")
        v_id = self.getVideoId(video_url)
        workdir = self.createWorkDir(v_id, outputFolder)
        cam = cv2.VideoCapture(self.bestVideo)
        if not cam.isOpened():
            raise IOError('ExtractFrames Can\'t open "Yolo2Model"')
        frame_num = 0
        for i in my_progress_bar.progressbar(range(self.totalFrames), widgets=widgets):
        #while True:
            ret, frame = cam.read()
            if ret:
                blur_prediction = 0
                if (frame_num in frame_series):
                    blur_prediction = self.estimate_blur(frame)
                    cv2.imwrite(f'{outputFolder}/{v_id}/localMaxFrame_{frame_num}_{blur_prediction}.jpg', frame)
                frame_info = f'Frame: {frame_num}, Score: {blur_prediction}'
                cv2.putText(frame, frame_info, (10, frame.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                key = cv2.waitKey(1) & 0xFF
                # if key == ord('q'): break
                if key == ord('s'): cv2.imwrite(f'frame_{time.time()}.jpg', frame)
                frame_num += 1
        cam.release()
        print('Frames compared:     ' + str(frame_num))
        return workdir


             
    def findBestFrame(self, frames, average_cut):
        max_colorfullness = average_cut
        max_score = 0.5
        if len(frames) == 0:
            return None
        best_frame = frames[0]
        for frame in frames:
            if int(frame['blur']) > max_colorfullness:
                max_colorfullness = int(frame['blur'])
                best_frame = frame
        for frame in frames:
            if int(frame['blur']) > average_cut:
                if (len(frame['predictions']) > 0):
                    max_area = 0
                    predictions = frame['predictions']                     
                    for prediction in predictions:
                        if (float(prediction['score']) > max_score):
                            if(int(prediction['area']) > max_area):
                                max_area = int(prediction['area'])
                                best_frame = frame
        return best_frame
    
    
    
    

    def getColorResults(self, workDir, average_cut):
        with open(f'{workDir}/metadata.json') as f:
            metadata = json.load(f)
        scenes = dict()
        for x in metadata:
            scenes[x['scene']] = scenes[x['scene']] + [x] if x['scene'] in scenes.keys() else [x]
        scenes = dict(sorted(scenes.items()))
        best_frames = []
        for scene, frames in scenes.items():
            best_frame = self.findBestFrame(frames, average_cut)
            # if self.log: 
            #     print(f'************** Best frame: scene {scene}\n{best_frame}')
            if best_frame is not None: 
                best_frames.append(best_frame)
        best_frames = self.getOnlyNFrames(best_frames)
        with open(f'{workDir}/filtered_metadata.json', 'w') as f:
            json.dump(best_frames, f, indent=4, separators=(',', ': '), sort_keys=True)
        for file in best_frames:
            fname = self.path_leaf(file['file'])
            new_name = f'{workDir}/selected_{fname.split("_")[1]}.jpg'
            os.rename(file['file'], new_name)
        images = glob.glob(f'{workDir}/*')
        for filename in images:
            if ('\localMaxFrame' in filename) or filename.endswith('.json'):
                os.unlink(filename)
        return best_frames
    
    
    
    
    def getOnlyNFrames(self, best_frames):
        while len(best_frames) > self.n_max_frames:
            min = int(best_frames[0]['blur'])
            frame_idx, counter = 0, 0
            for frame in best_frames:
                if int(frame['blur']) < int(min):
                    frame_idx = counter
                    min = frame['blur']
                counter+=1
            del best_frames[frame_idx]
        return best_frames
    
    
    
    
    
    def predictFaces(self, image):
        #if self.log: print('predictFaces: ', image)
        # Load the cascade
        #if self.log: print('face_cascade')
        # Convert into grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        predictions = []
        #add the faces
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
    
    
    
   
    def processVideo(self, url, outputFolder):
        serie, metadata = self.getImgseries(url)   
        tmp = metadata    
        average_cut = np.average(serie)
        workdir = self.extractFrames(url, serie, outputFolder)
        print("-- Extract frame ok")
        metadata = self.batchHistCompare(workdir, metadata, corr=0.5)
        #get at least 5 scenes by improvement of frame correlation
        incr = 0.1
        # if log: print(metadata)
        size = int(metadata[len(metadata)-1]['scene']) + 1 if len(metadata)>0 else 0
        if size < self.n_max_frames:
            #lastScene_index = self.n_max_frames
            print(f'There are less than {self.n_max_frames} scenes. Increasing correlation threshold')
        while size < self.n_max_frames:
            metadata = self.batchHistCompare(workdir, tmp, corr=0.5+incr)
            incr += 0.1
            size = int(metadata[len(metadata)-1]['scene']) + 1 if len(metadata)>0 else 0
            if incr == 0.6:
                break
        print(f"-- Selected metadata length: {len(metadata)}")
        with open(f'{workdir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4, separators=(',', ': '), sort_keys=True)
        return self.getColorResults(workdir, average_cut)




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
                
                
                
                
                
                
                
                
                
                
                
                
                
                
 