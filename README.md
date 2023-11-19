# Jukebeat

Code for [paper](http://ismir2023program.ismir.net/lbd_363.html) "Beat and Downbeat Tracking with Generative Embeddings" in Late Breaking Demo in the International Society for Music Information Retrieval Conference, ISMIR, 2023.

## Setup

Jukebeat is developed with Python 3.8.2, CUDA 11.2, cuDNN 8.1.0, and FFmpeg 4.2.2.  
It is recommend to create a virtual environment:
```
python -m venv jukebeat
source jukebeat/bin/activate
```
To install dependencies:
```
pip install -r requirements.txt
```

## Data

To download the audio data of Ballroom, Hainsworth, SMC, and GTZAN, visit [here](https://github.com/zhaojw1998/Beat-Transformer#audio-data). The download [link](https://ddmal.music.mcgill.ca/breakscience/dbeat/) for HJDB audio is currently not accessible. To download the beat and downbeat annotations, visit [here](https://github.com/superbock/ISMIR2019).

To reproduce the experiments, please organize the data as follows:

```
data
└───ballroom
	└───ballroomData
		│	ChaChaCha
			│	...		audio files
		│	...			other subgenre folders
	└───ballroomAnnotations
		│	...			annotation files
└───hainsworth
	└───hainsworthData
		│	...			audio files
	└───hainsworthAnnotations
		│	...			annotation files
└───smc
	└───smcData
		│	...			audio files
	└───smcAnnotations
		│	...			annotation files
└───hjdb
	└───hjdbData
		│	...			audio files
	└───hjdbAnnotations
		│	...			annotation files
└───gtzan
	└───gtzanData
		│	blues
			│	...		audio files
		│	...			other subgenre folders
	└───gtzanAnnotations
		│	...			annotation files
```
Please be reminded to change the `DATA_PATH` parameter in `code/paras.py` to your own path.

To compute jukebox embeddings for a dataset (e.g., ballroom):
```
python code/generate_jukebox_emb.py --dataset ballroom
```
The embeddings are stored in the `<dataset_name>JukeboxAvePool` folder under the corresponding dataset directory.

## Train
To train beat and downbeat tracking models for fold 0 (8 folds in total for the 8-fold cross validation):
```
python code/main.py --gpu 0 --fold 0 --input_repr audio
python code/main.py --gpu 0 --fold 0 --input_repr jukebox
python code/main.py --gpu 0 --fold 0 --input_repr audio --augmentation
python code/main.py --gpu 0 --fold 0 --input_repr jukebox --augmentation
```