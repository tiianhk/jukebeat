import sys
import os
import tensorflow as tf
import numpy as np
from scipy.ndimage import maximum_filter1d
from load_data import get_tracks
from data_sequence import DataSequence
from preprocess import PreProcessor
from evaluate import evaluate_beats, evaluate_downbeats, predict
from postprocess import beat_tracker, downbeat_tracker, bar_tracker
from paras import FPS, DATA_PATH

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    output_dir = sys.argv[1]
    gpu_id = sys.argv[2]
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    input_repr = output_dir[:output_dir.find('_')]
    path = os.path.join(DATA_PATH, f'../exp/{output_dir}') # includes 8-fold CV experiments.
    dirs = os.listdir(path)
    
    tracks = get_tracks(['gtzan'], input_repr)

    # these tracks don't have downbeat annotation
    key_exceptions = ['gtzan_jazz.00003',
                      'gtzan_jazz.00009',
                      'gtzan_jazz.00010',
                      'gtzan_jazz.00014',
                      'gtzan_jazz.00018',
                      'gtzan_jazz.00020']

    # adding all genres in gtzan
    genres = []
    for t in tracks.values():
        if t.genre not in genres:
            genres.append(t.genre)
    
    ds = {}
    print('wrapping gtzan data..')

    for genre in genres:
        print(f'wrapping data for {genre}..')
        ds[genre] = DataSequence({k:v for k,v in tracks.items() if v.genre == genre},
                                        PreProcessor(), pad_frames=2)

    # adding full as a genre, that is, all genre combined
    print(f'wrapping the full dataset..')
    ds['full'] = DataSequence(tracks, PreProcessor(), pad_frames=2)

    # loading 8 models from the 8-fold CV experiments.
    models = [tf.keras.models.load_model(f'{path}/{dir}/model_final.h5', compile=False)\
                                         for dir in dirs if os.path.isdir(f'{path}/{dir}')]

    # eval
    for genre, dataset in ds.items():
        print(f'evaluating on gtzan {genre}..')
        acts = []

        # inference
        for i, model in enumerate(models):
            print(f'model {i} inference..')
            act, _ = predict(model, dataset)
            acts.append(act)
        beat_detections = {}
        downbeat_detections = {}
        bar_detections = {}
        print('postprocessing..')
        
        db_act = {}

        # get prediections
        for key in acts[0].keys():
            beats_act = np.mean(np.stack([act[key]['beats'] for act in acts]), axis=0)
            downbeats_act = np.mean(np.stack([act[key]['downbeats'] for act in acts]), axis=0)
            combined_act = np.mean(np.stack([act[key]['combined'] for act in acts]), axis=0)
            db_act[key] = downbeats_act
            beats = beat_tracker(beats_act)
            downbeats = downbeat_tracker(combined_act)
            beat_idx = (beats * FPS).astype(np.int_)
            bar_act = maximum_filter1d(downbeats_act, size=3)
            bar_act = bar_act[beat_idx]
            bar_act = np.vstack((beats, bar_act)).T
            try:
                bars = bar_tracker(bar_act)
            except IndexError:
                bars = np.empty((0, 2))
            beat_detections[key] = beats
            downbeat_detections[key] = downbeats
            bar_detections[key] = bars

        # load annotations
        if genre == 'full':
            beat_annotations = {k: v.beats.times for k, v in tracks.items()}
            downbeat_annotations = {k: v.beats.times[v.beats.positions == 1] for k, v in tracks.items()\
                                    if k not in key_exceptions}
        else:
            beat_annotations = {k: v.beats.times for k, v in tracks.items() \
                                if v.genre == genre}
            downbeat_annotations = {k: v.beats.times[v.beats.positions == 1] for k, v in tracks.items() \
                                    if v.genre == genre and k not in key_exceptions}

        # eval and write file
        print(f'writing the results to file..')
        resdir = f'{path}/gtzan-{genre}-eval.txt'
        with open(resdir,'a') as f:
            f.write('Beat evaluation\n---------------\n Beat tracker:    ')
            f.write(evaluate_beats(beat_detections, beat_annotations).tostring())
            f.write('\n Downbeat tracker:    ')
            f.write(evaluate_beats(downbeat_detections, beat_annotations).tostring())
            f.write('\nDownbeat evaluation\n-------------------\n  Bar tracker:     ')
            f.write(evaluate_downbeats(bar_detections, downbeat_annotations).tostring())
            f.write('\n Downbeat tracker:    ')
            f.write(evaluate_downbeats(downbeat_detections, downbeat_annotations).tostring())
            f.write('\n')
