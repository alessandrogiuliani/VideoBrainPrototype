
# VideoBrain Prototype

## Introduction

The prototype is entirely implemented in Python language. It embeds several packages, be sure to have installed all dependencies, as reported in the following section. The prototype has been tested in a specific Linux-based environment. See the section “Virtual Machine” for all info. 

Latest release changes: 
- Close up face recognition added
- Now language models are pre-loaded simultaneously 



---

## Installation

#### By script
The script installs both the environment and the prototype in a Linux-based OS. The system is tested only for Ubuntu20.10; to avoid possible installation issues in different OS, the prototype should be installed manually (see next section). 

1. Download in your desired folder the bash script ```install_environment.sh``` from [here](https://drive.google.com/open?id=1B_QCINqF0wsL8SERLvZeYtuoP_FhnhXd)
2. Open the  terminal, and launch the command
	```
	sh install_environment.sh
	```
	 The scripts will install the Python environment (including Anaconda IDE and tools) .  
	 ***NOTE***: the other scripts needed for installing the prototype are automatically download. No additional manual downloads are required.
3. Close the terminal and relaunch it (this is needed for automatically activating the environment)
4. Launch the script ```install_prototype```:
```
		sh install_prototype.sh
```
5. The system is now ready, the prototype could be launch with the command 
	```
			sh RunVideoBrain.sh
	```
	 or by the command
	```
		    python [path]/web_processor.py
	```
	replacing  [path] with the path folder where the prototype is installed.

#### Manual installation
1. Download the prototype from [here](https://drive.google.com/open?id=1wndxGkLnA_02ob2awirzlpOSsnq6KAP2).
2. Extract the source files contained in `Prototype.rar` in the desired folder. There is no need for further actions, but the Python packages reported below should be installed. Note that the code is implemented and tested for Python version **3.8.5**. Be sure to run Python under the right environment.
3. Install the following packages in the Python environment. Each package of the following list is annotated with the release version used for local testing.

	- opencv-python 4.4.0.46
	- youtube-dl 2020.11.18
	- keras 2.4.3
	- tensorflow 2.3.0
	- flask 1.1.1
	- flask_cors 3.0.9
	- nltk 3.5
	- pandas 1.1.4
	- numpy 1.18.1
	- scipy 1.5.2
	- gensim 3.8.3
	- pytrends 4.7.3
	- pafy 0.5.5
	- Pillow 8.0.1
	- matplotlib 3.3.3
	- python_utils 2.4.0
	- scikit-learn 0.23.2
	- beautifulsoup4 4.9.3
	- flask_restplus 0.13.0
	- h5py 2.10.0
	
	or, from Python shell, launch the following command (be sure `pip` package is installed):
	
	```
	pip install opencv-python==4.4.0.46 youtube-dl==2020.11.18 keras==2.4.3 tensorflow==2.3.0 flask==1.1.1 flask_cors==3.0.9 nltk==3.5 pandas==1.1.4 numpy==1.18.1 scipy==1.5.2 gensim==3.8.3 pytrends==4.7.3 pafy==0.5.5 Pillow==8.0.1 matplotlib==3.3.3 python_utils==2.4.0 scikit-learn==0.23.2 beautifulsoup4==4.9.3 flask_restplus==0.13.0 h5py==2.10.0
	```
4. Download the embedding vectors in the folder `model_data`:
	- For the *English* model, the embeddings could be downloaded from [here](https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz).  
**Important**: place the file without any other actions. No unzip is needed.

	-   For the *Italian* model, the embeddings could be downloaded from [here](https://www.dropbox.com/s/orqfu6mb9cj9ewr/it.tar.gz?dl=0)  
	**Important**: unzip the file content in the same folder. The model file is `it.bin`.

## Configuration
Framework settings and default algorithm parameters are loaded from the file `config.ini`, placed in the folder `config`.

 ### Config.ini
All configuration parameters are reported below. *Framework  parameters* are the general prototype settings; if there is the need to change them, the new values should be placed here, and the application needs to be restarted. *Main parameters*, *Thumbnail generation*, and *Tag generation* are the default values for each algorithm.

#### Framework parameters

-   ***PORT_NUMBER***: the port of the webserver.
-   ***STATIC_URL_PATH***: the path of the static files folder (CSS, Js, etc.).
-   ***LOG***: selects if printing (in the Python console) the various message logs during the execution. Type: boolean
-   ***load_embedding_model***: select if loading the embedding models. If True, all implemented language embeddings (see language parameter) are loaded. Type: boolean
-   ***luminati_username***: the user ID of Luminati proxy service. It is needed to avoid the YouTube temporary ban (it happens when too many requests are sent to the YouTube portal).
    
-  ***luminati_password***: the password of Luminati proxy service.
    

#### Main parameters

-   ***domain***: the default domain of the video. Type: string. Although it is recommended to always specify the domain, this default value could be useful when using the app only in a specific domain (*). Currently, the following domains have been implemented:  
	-  music
	-  sport
	-  cars
	-  food
	-  animals
	-  tech

	(*) For more info, check the section “Featured domains” at the end of this document.
-   ***generate_thumbnails***: selects to compute the thumbnail generation. Type: boolean    
-   ***generate_tags***: select to compute the tag generation. This is possible ONLY if LOAD_EMBEDDING_MODEL is set to True. Otherwise, no tags would be generated. Type: boolean
    
#### Thumbnail Generation

-   ***output_folder_thumbnails***: the main folder where all generated thumbnails will be saved. Note: for each video, thumbnails will be stored in a subfolder named as the video ID. 
-   ***n_max_frames***: the number of generated thumbnails.  
-   ***method***: the selected algorithm for thumbnail generation. The values correspond to the following algorithms (we named each algorithm as reported in the WEBIST paper):
	-   BFP: Blur-based Frame Pruning
	-   CFP: Colorfulness-based Frame Pruning
	-   DOD: Dynamic Object Detection
	-   FSI: Fast Scene Identification

-   ***corr_threshold***: the correlation threshold used for selecting frames in the algorithms BFP, CFP, and DOD (the threshold being the value of the correlation of 2 images).
-   ***fsi_threshold***: the threshold used for identifying scene changes (the threshold being the value of the difference between the HSV values of 2 images).   
-   ***process_faces***: select if predicting faces. If the value is False, Yolo will be adopted to recognize objects in images. We suggest setting the value as False and selecting True only for a single run, as faces are helpful mainly for the “music” domain. In contrast, other domains should rely on YOLO analysis.
-   ***smile_detection***: select if predicting smiles in detected faces. If the value is True, the smile detector of OpenCV will be adopted.
-   ***open_eye_detection***: determine if predicting open eyes in detected faces. If the value is True, the open-eyes detector of OpenCV will be adopted.
-   ***close_up_ratio***: if face prediction is selected, this parameter permits the identification of close-up faces. In particular, it sets the minimum size of the face's bounding box to be recognized in terms of image ratio. The parameter ranges from 0 to 1: the more the value is, the larger the face will be recognized. Setting a high value of `close_up_ratio` permits to identify only close-up faces.   
***Example***: setting `close_up_ratio = 0.8` means that the systems would identify only faces bounded with a box sized at least 80% of the original frame size.
-   ***max_length***: the maximum video length (in seconds), meaning that only the first max_length seconds of a video will be analyzed. Cutting long videos is helpful to save computational resources. If the value is 0, the entire video will be downloaded and analyzed.

    

#### Tag generation

-   ***output_folder_tags***: the main folder where all generated tags will be saved. Note: for each video, thumbnails will be stored in a subfolder named as the video ID.  
-   ***language***: the language of the video or channel.  
    **Note**: currently, only English and Italian languages have been implemented. 
-   ***n_suggested_tags*** : the number of output generated tags. Note: output tags are the most related hot trends
-   ***granularity***: the granularity level in the tags-trend similarity computation. Values:

	-   WL: word level. Each trend is compared to each original tag.    
	-   SL: sentence level. Each trend is compared to the entire original tags list. 
	-   CL: cluster level. The original tags are clustered, the trend at hand being compared to each cluster.

-   ***get_original_tags***: if the value is True, the original tags will be taken as input features. 
-   ***get_title***: if the value is True, the video title will be taken as an additional input feature.
-   ***get_description***: if the value is True, the video description will be taken as an additional input feature.  
-   ***rising_trends***: if the value is True, also the “rising” trends of a given topic will be retrieved. If it is False, only “top” trends will be retrieved. 
***Note***: at least one among title, description, and original tags must be selected to let the prototype work properly.

## Execution
To start the server, launch the Python script `web_processor.py`

    python web_processor.py

When the server is running, the prototype could analyze a video by means of a URL string in which all parameters may be set. The URL will be formatted in the following way:

	hosturl:port/api?parameters

After the symbol “?”, each parameter, separated from others with the char “&”, is in the form *parameter=value*. If a parameter is not explicitly specified, the default value will be loaded from the `config.ini` file.
  
Example:  

    localhost:8080/api?id=4VMBuAPzhKY&domain=music&nframes=5&gen_tags=true&ntags=5

### URL parameters
-   **id**: the YouTube video ID. (mandatory)
-   **domain**: the domain of the video. As reported in the previous section, we implemented the domains *music, sport, cars, food, animals, tech* 
-   **gen_thumb**: select to compute the thumbnail generation. Values: *True/False*    
-   **gen_tags**: select to compute the tag generation. Values:  *True/False*    
-   **nframes**: the number of generated thumbnails.
-   **method**: the selected algorithm for thumbnail generation. Values: *BFP, CFP, DOD, FSI*
-   **cth**: the correlation threshold used for selecting frames in the algorithms BFP, CFP, and DOD (the threshold is the value of the correlation of 2 images). 
-   **fth**: the threshold used for identifying scene changes (the threshold being the value of the difference between the HSV values of 2 images).
-   **faces**: select if predicting faces. Values:  *True/False*    
-   **smiles**: select if predicting smiles in detected faces. Values:  *True/False*    
-   **open_eyes**: select if predicting open eyes in detected faces.  *True/False*
-   ***close_up_ratio***: it sets the minimum size of the face's bounding box to be recognized in terms of image ratio. Values:  *[0.0, 1.0]*  
-   **max_length**: the maximum video length (in seconds) to be analyzed. Cutting long videos is useful to save computational resources. If the value is 0, the entire video will be downloaded and analyzed.
-   **ntags**: the number of output generated tags.
-   **lang**: the language of the video metadata. Values: *italian, english*
-   **gran**: the granularity level in the tags-trend similarity computation. Values: *WL, SL, CL* 
-   **get_original_tags**: original tags selection. Values: True/False
-   **get_title**: video title selection. Values: *True/False*    
-   **get_description**: video description selection. Values:  *True/False*    
-   **rising_trends**: “rising” trends selection. Values:  *True/False*    

## Featured Domains

For the sake of completeness, this section reports the main information about the available domains, useful to understand the behavior of the prototype for thumbnail generation.  
Depending on the selected domain, the system will perform the elaboration in different ways. In details:

-   Domain **music**: the system automatically will recognize frames containing faces (note that the face predictor may detect false positives, being frames without faces). For this purpose, face recognition of OpenCV is adopted.
-   For the other domains, the prototype will recognize frames containing related objects employing YOLO, an object detection framework embedded in the prototype. YOLO is pre-trained with COCO dataset. Depending on the domain, the framework will recognize the following objects:
	-   **food**: bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake;
	-   **cars**: car, motorbike, truck, bus;
	-   **tech**: tvmonitor, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster;
	-   **animals**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe;
	-   **sport**: bicycle, car, motorbike, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, person.
	
## Future implementations

- **Language models** Further langua models might be included. Embedding vectors may be found [here]( https://github.com/Kyubyong/wordvectors) or [here](https://fasttext.cc/docs/en/crawl-vectors.html)
- **Integration of further domains** Further domains may be implemented. Google Trends categories may be chosen from [here](https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories) 