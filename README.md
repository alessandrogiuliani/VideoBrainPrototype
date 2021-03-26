
# VideoBrain Prototype

## Introduction

The prototype is entirely implemented in Python language. It embeds several packages, be sure to have installed all dependencies, as reported in the following section. The prototype has been tested in a specific Linux-based environment. See the section “Virtual Machine” for all info. 

Latest release changes: 
- Tag generation: added rising trends retrieval (in combination with top trends)
- Tag generation: added the option of considering video title and description. 



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
