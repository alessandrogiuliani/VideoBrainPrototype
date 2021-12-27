# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:42:30 2020

@author: Alessandro Giuliani, Eugenio Gaeta, Leonardo Piano

"""
from yolo import YOLO
from filters import smooth
from PIL import Image
from six.moves.urllib.parse import urlparse
import glob
import ntpath
import logging
import os
import cv2
import time
import logging.config
import my_pafy as pafy
import numpy as np
import json
from scipy import signal
import my_progress_bar

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil

widgets=[
    ' [', my_progress_bar.Percentage(), '] ',
    my_progress_bar.Bar(marker='█'),
    ' (', my_progress_bar.ETA(), ') ',
]
yoloInstance = YOLO()

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()


#*****************************************************************************
#***********************   General Algorithm  ********************************
#*****************************************************************************


class GenericThumbnailProcessor(object):


    def __init__(self, **kwargs):
        self.filteredFrames = 50
        self.domain = kwargs.get('domain', None)
        self.n_max_frames = kwargs.get('n_max_frames', 5)
        self.log = kwargs.get('log', False)
        self.corr_threshold = kwargs.get('corr_threshold', 0.5)
        self.process_color = kwargs.get('process_color', True)
        self.process_faces = kwargs.get('process_faces', True)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        self.eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smiles = kwargs.get('smiles', True) 
        self.open_eyes = kwargs.get('open_eyes', True)
        self.max_length = kwargs.get('max_length', 0)
        self.opener = kwargs.get('opener', None)
        self.close_up_ratio = kwargs.get('close_up_ratio', 0.1)





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
                vcap = cv2.VideoCapture(s.url)
                if not vcap.isOpened():                   
                    continue
                vcap.release()
                width = s.dimensions[0]
                if width <= width_limit:
                    if res is None:
                        res = s
                        continue
                    if  width > res.dimensions[0]: res = s
        return res.url
    
    
    
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




    #analize the laplacian fame series
    def extractFramesFromSeries(self, serie):
        if self.log: print("extractFramesFromSeries")
        f = smooth(np.asarray(serie))
        s_av = np.average(f)
        f = f-s_av
        df = smooth(np.gradient(f))
        df[df < 0] = 0
        y_coordinates = df # convert your 1-D array to a numpy array if it's not, otherwise omit this line
        #average of derivative of blur
        average = np.average(y_coordinates)
        df[df < average] = 0
        if self.log: print("df average: "+str(average))
        max_peak_width = average
        if (max_peak_width<2):
            max_peak_width = 2
        peak_widths = np.arange(1, max_peak_width)
        if self.log: print("peak detection on average of derivative: "+str(peak_widths))
        peak_indices = signal.find_peaks_cwt(y_coordinates, peak_widths)
        peak_count = len(peak_indices) # the number of peaks in the array
        if self.log:
            print ("df average: "+str(average))
            print (peak_indices)
            print ("peak before filtering: "+str(peak_count))
        #array of zeros of lenght of derivative in order to filter peaks
        array_of_zeros = np.zeros(len(df),dtype=bool)
        i = 0
        #set to true the indexes that corresponds to a peak
        for x in np.nditer(array_of_zeros):
            array_of_zeros[i] = (i in peak_indices)
            i = i + 1
        #compress the peaks
        array_peaks = df[array_of_zeros==True]
        #compute the average of the peaks compressed
        if self.log: print ("Peaks array len: "+str(len(array_peaks)))
        average = np.average(array_peaks)
        if self.log: print ("average Peaks array : "+str(average))
        max_peak_width = average
        if (max_peak_width<2):
            max_peak_width = 2
        peak_widths = np.arange(1, max_peak_width)
        if self.log: print("peak detection on average of derivative: "+str(peak_widths))
        peak_count = len(peak_indices) # the number of peaks in the array
        array_of_zeros = np.zeros(len(df),dtype=bool)
        if self.log: print ("array_of_zeros: "+str(len(array_of_zeros)))
        try:
            f_count = 0
            incr = 1
            while True:
                i = 0
                #set to true the indexes that corresponds to a peak
                max = np.amax(f)
                if self.log: print ("max: "+str(max))
                s_av = max - incr/10*max
                if self.log: print ("cut off: "+str(s_av))
                f2 = f-s_av
                if self.log: print ("Array of zezo len: "+str(len(array_of_zeros)))
                if self.log: print ("f2 len: "+str(len(f2)))
                for x in np.nditer(array_of_zeros):
                    try:
                        array_of_zeros[i] = (i in peak_indices) and (f2[i]>0)
                        i = i + 1
                    except Exception as e: 
                        if self.log: print(e)
                        i = i + 1
                        if self.log: print("Cannot filter the array of zeros error in for")
    #            plt.figure(1)
    #            plt.plot(array_of_zeros*100) 
                f_count = np.count_nonzero(array_of_zeros)
                incr = incr + 1
                #print ("f_count: "+str(f_count))
                if f_count >= self.filteredFrames or incr==10:
                    break
        except Exception as e: 
            print(e)
            print("Cannot filter the array of zeros")
        if self.log: print ('Average: '+str(average))
        if self.log: print ("local max found: "+str(peak_count))
        if self.log: print ("Remove peak arount the average: "+str(np.count_nonzero(array_of_zeros)))
        r = np.empty(np.count_nonzero(array_of_zeros))
        i = 0
        j = 0
        for x in np.nditer(peak_indices):
            if(array_of_zeros[peak_indices[i]]):
                if(j<np.count_nonzero(array_of_zeros)):
                    r[j]=peak_indices[i]
                    j = j + 1
            i = i + 1
        if self.log:
            print ("Remove peak around the average: "+str(len(r)))
            print (r)
        return r




    def estimate_blur(self, image, threshold=100):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_map = cv2.Laplacian(image, cv2.CV_64F)
        score = np.var(blur_map)
        return score, bool(score < threshold)
        



    def compareHist(self, file1, file2, stat=cv2.HISTCMP_CORREL):
        target_im = cv2.imread(file1)
        target_hist = cv2.calcHist([target_im], [0], None, [256], [0, 256])
        comparing_im = cv2.imread(file2)
        comparing_hist = cv2.calcHist([comparing_im], [0], None, [256], [0, 256])
        diff = cv2.compareHist(target_hist, comparing_hist, stat)
        return diff




    def batchHistCompare(self, dir,  corr=0.99):
        f_dist = corr
        metadata = []
        if dir: videos = [dir]
        for video in videos:
            try:
                logger.info(video)
                video_id = self.path_leaf(video)
                files = glob.glob(video+'/*')
                files.sort(key=os.path.getmtime)
                count = 0
                scene_counter = 0
                target = files[0]
                for file in files:
                    try:

                        fname = self.path_leaf(file)
                        if self.log: print(file)
                        dist = self.compareHist(target,file)
                        target = file
                        isChanged = " no scene change."
                        if (dist<f_dist): 
                            isChanged = " THIS IS A NEW SCENE."
                            scene_counter = scene_counter +1
                        if self.log: print(f'Image {count}, {self.path_leaf(file)} compare {dist}. {isChanged}')
                        metainfo = {}
                        metainfo['file'] = file
                        metainfo['frame'] = fname.split('_')[0]
                        metainfo['blur'] = fname.split('_')[1].split('.')[0]
                        metainfo['scene'] = scene_counter
                        metainfo['corr'] = dist
                        try:
                            image = Image.open(file)
                            if self.log: print('Image.opened: '+file)
                        except:
                            print('File Open Error! Try again!')
                        if self.log: print('Try to make a prediction on faces.')
                        if self.domain=='music':
                            predictions = self.predictFaces(file)
                            metainfo['predictions'] = predictions['predictions']
                        else:                            
                            metainfo['predictions'] = yoloInstance.detect_img(image, self.domain)
                            if self.process_faces:
                                predictions = self.predictFaces(file)
                                metainfo['predictions'] += predictions['predictions']
                        metadata.append(metainfo)
                        count = count + 1
                    except:
                        logger.error(f'video id {video_id} not processed image {file}.', exc_info=True)
            except:
                logger.error(f'video id {video_id} not processed url {video}.')
        #print(metadata)
        return metadata
        #this function draw the boxes around the detected objects



    
    def processColorAndBlur(self, video_url,v=''):
        if self.log: print("processColorAndBlur")
        colorfulness_series = []
        blur_series = []
        video = video_url
        if self.log: print("video_url "+str(video_url))
        #it returns the id if is the url of a youtube video
        v_id = self.getVideoId(video)
        if self.log: print("v_id "+str(v_id))
        if self.is_url(video):
            if self.log: print("before pafy")
            if self.opener is not None: self.opener.open(video)
            videoPafy = pafy.new(video)
            #video = videoPafy.getbest(preftype="mp4").url
            if self.log: print("after pafy")
            video = self.getBestVideo(videoPafy)
            if self.opener is not None: self.opener.close()
        else:
            v_id = v
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
                    self.totalFrames = frame_num 
                    if self.log: print('Can\'t read video data. Potential end of stream')
                    return blur_series,colorfulness_series
                blur_prediction = self.estimate_blur(frame)[0]#blur score
                colorfulness_prediction = self.image_colorfulness(frame)
                blur_series.append(blur_prediction)
                colorfulness_series.append(colorfulness_prediction)
                frame_num += 1
                if self.max_length > 0:
                    if frame_num > frame_limit:
                        cam.release()
                        self.totalFrames = frame_num 
                        return blur_series,colorfulness_series
        finally:
            cam.release()



    
    def extractFrames(self, video_url,frame_series, outputFolder, v=''):
        if self.log: print("extractFrames ok")
        video = video_url
        #it returns the id if is the url of a youtube video
        v_id = self.getVideoId(video)
        if self.log: print("extractFrames video: "+video)
        if self.is_url(video):
            if self.opener is not None: self.opener.open(video)
            videoPafy = pafy.new(video)
            #video = videoPafy.getbest(preftype="mp4").url
            video = self.getBestVideo(videoPafy)
            if self.opener is not None: self.opener.close()
        else:
            v_id = v    
        workdir = self.createWorkDir(v_id, outputFolder)
        cam = cv2.VideoCapture(video)
        if not cam.isOpened():
            raise IOError('ExtractFrames Can\'t open "Yolo2Model"')
        frame_num = 0
        fps = cam.get(cv2.CAP_PROP_FPS)
        frame_limit = fps * self.max_length 
        for i in my_progress_bar.progressbar(range(self.totalFrames), widgets=widgets):
            ret, frame = cam.read()
            blur_prediction = 0
            if (frame_num in frame_series) == True: 
                blur_prediction = self.estimate_blur(frame)[0]#blur frames
                cv2.imwrite(f'{outputFolder}/{v_id}/localMaxFrame_{frame_num}.jpg', frame)
                if self.log: print(f'{outputFolder}/{v_id}/localMaxFrame_{frame_num}.jpg')
            # Draw additional info
            frame_info = f'Frame: {frame_num}, FPS: {fps:.2f}, Score: {blur_prediction}'
            cv2.putText(frame, frame_info, (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            key = cv2.waitKey(1) & 0xFF
            # Exit
            if key == ord('q'):
                break
            # Take screenshot
            if key == ord('s'):
                cv2.imwrite(f'frame_{time.time()}.jpg', frame)
            frame_num += 1
        return workdir




        
        
    def getColorResults(self, workDir, average_cut):
        workDir=workDir+'/'
        with open(workDir+'metadata.json') as f:
             metadata = json.load(f)
        if len(metadata) == 0:
             images = glob.glob(workDir+'*')
             for filename in images:
                 if ('localMaxFrame' in filename) or filename.endswith('.json'):
                     os.unlink(filename)
                     print('No relevant frames found. Try another method or change parameters.')
                     return 
        lastScene = metadata[len(metadata)-1]
        to = int(lastScene['scene'])
        scenes = []
        for i in range(to+1):
            scenes.append([])
        for data in metadata:
            scenes[data['scene']].append(data)
        if self.log:
            print("***************************")
            print(scenes)
        #best frame for scene max object area if blur > 1000
        best_frames = []
        for i in range(to+1):
            if self.log: print("*********** or i in range(to  ****************")
            frames = scenes[i]
            max_colorfullness = average_cut
            ref_colorfullness = average_cut
            max_score = 0.5
            best_frame = None
            if (len(scenes[i])>0):
                best_frame = scenes[i][0]
            if self.log: print("*************BEST FRAME**************")
            for frame in frames:
                if (int(frame['blur'])>max_colorfullness):
                    max_colorfullness = int(frame['blur'])
                    best_frame = frame
                    if self.log:
                        print("*************BEST FRAME**************")
                        print(best_frame)
            for frame in frames:
                if (int(frame['blur'])>ref_colorfullness):
                    #seach for max object score 
                    if (len(frame['predictions'])>0):
                        max_area = 0
                        predictions = frame['predictions']
                        for prediction in predictions:
                            if (float(prediction['score'])>max_score):
                                if(int(prediction['area'])>max_area):
                                    max_area = int(prediction['area'])
                                    best_frame = frame
            if self.log:
                print("************** Best frame i="+str(i))
                print(best_frame)
            if best_frame is not None: 
                best_frames.append(best_frame)
        if self.log:
            print("*************************************")
            print(best_frames)
        best_frames = self.getOnlyNFrames(best_frames)
        with open(workDir + '/filtered_metadata.json', 'w') as f:
            json.dump(best_frames, f, indent=4, separators=(',', ': '), sort_keys=True)
        for file in best_frames:
            fname = self.path_leaf(file['file'])
            new_name = workDir+fname.replace('localMaxFrame', 'finalThumb')
            os.rename(file['file'], new_name)
        images = glob.glob(workDir+'*')
        for filename in images:
            if ('localMaxFrame' in filename) or filename.endswith('.json'):
                os.unlink(filename)
        return best_frames
    
    
    
    def getOnlyNFrames(self, best_frames):
        if self.log:
            print("*************************************")
            print(best_frames)
        while len(best_frames) > self.n_max_frames:
            #remove min
            min = int(best_frames[0]['blur'])
            frame_idx = 0
            counter = 0
            for frame in best_frames:
               
                if int(frame['blur'])<int(min):
                    frame_idx = counter
                    min = frame['blur']
                counter+=1
               
            del best_frames[frame_idx]
        return best_frames


    #use
    def processVideo(self, video_url, outputFolder):
        self.opener.open(video_url)
        if self.log: print("Processing at: "+os.getcwd())
        blur_series, colorfulness_series = self.processColorAndBlur(video_url)
        print("-- Blur colorfulness series ok")
        if self.process_color:
            serie = colorfulness_series
        else:
            serie = blur_series
        average_cut = np.average(serie)
        frame_series = self.extractFramesFromSeries(serie)
        print("-- Frame series ok")
        workdir = self.extractFrames(video_url, frame_series, outputFolder)
        print("-- Extract frame ok")
        metadata = self.batchHistCompare(workdir, corr = self.corr_threshold)
        #get at least 5 scenes by improvement of frame correlation
        incr = 0.1
        if self.log: print(metadata)
        lastScene_index = int(metadata[len(metadata)-1]['scene']) if len(metadata)>0 else 0
        
        if(len(metadata) < self.n_max_frames):
            lastScene_index = self.n_max_frames
            print(f'There are less than {self.n_max_frames} scenes. Increasing correlation threshold')
        while lastScene_index < self.n_max_frames:
            metadata = self.batchHistCompare(workdir, corr=self.corr_threshold+incr)
            incr += 0.1
            lastScene_index = int(metadata[len(metadata)-1]['scene']) if len(metadata)>0 else 0
        if self.log: print("5 filter")
        print(f"-- Selected metadata length: {len(metadata)}")
        with open(workdir+'/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4, separators=(',', ': '), sort_keys=True)
            self.opener.close()
        return self.getColorResults(workdir, average_cut)
    
    
    
    def image_colorfulness(self, image):
          # split the image into its respective RGB components
    	(B, G, R) = cv2.split(image.astype("float"))
     
    	# compute rg = R - G
    	rg = np.absolute(R - G)
        
    	# compute yb = 0.5 * (R + G) - B
    	yb = np.absolute(0.5 * (R + G) - B)
     
    	# compute the mean and standard deviation of both `rg` and `yb`
    	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
    	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
     
    	# combine the mean and standard deviations
    	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
     
    	# derive the "colorfulness" metric and return it
    	return stdRoot + (0.3 * meanRoot)
    
    
    
    def predictFaces(self, imageFile):
        if self.log: print('predictFaces: ',imageFile)
        img = cv2.imread(imageFile)
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        window = (int(gray.shape[0]*self.close_up_ratio), int(gray.shape[1]*self.close_up_ratio))
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=window)
        predictions = {
                        "predictions": []
                        }
        # add the faces
        for (x, y, w, h) in faces:
            if self.smiles:
                roi_gray = gray[y:y + h, x:x + w]
                sm = self.smile_cascade.detectMultiScale(roi_gray, 1.1, 4)
                if len(sm) == 0:                   
                    continue
            if self.open_eyes:
                roi_gray = gray[y:y + h, x:x + w]
                eyes = self.eyes_cascade.detectMultiScale(roi_gray, 1.1, 4)
                if len(eyes) == 0: continue
            prediction = {"area": int(w*h),
                         "box": [int(x), int(y), int(w), int(h)],
                         "class": "face",
                         "score": 0.99}
            predictions['predictions'].append(prediction)
        #if self.log: print(json.dumps(predictions))
        return predictions


    
    
#*****************************************************************************
#**********************   Inheritance: BFP and CFP   *************************
#*****************************************************************************

class BFP(GenericThumbnailProcessor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.process_color = False




class CFP(GenericThumbnailProcessor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.process_color = True
