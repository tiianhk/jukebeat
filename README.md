# Jukebeat

Code for paper "Beat and Downbeat Tracking with Generative Embeddings" in Late Breaking Demo in the International Society for Music Information Retrieval Conference, ISMIR, 2023.

## Setup

## Data

The information for downloading the audio data of Ballroom, Hainsworth, SMC, GTZAN are shared by Jingwei Zhao [here](https://github.com/zhaojw1998/Beat-Transformer#audio-data). The download [link](https://ddmal.music.mcgill.ca/breakscience/dbeat/) for HJDB audio is currently not accessible. The beat and downbeat annotations are released by Sebastian Böck [here](https://github.com/superbock/ISMIR2019).

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
