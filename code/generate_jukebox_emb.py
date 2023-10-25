import librosa as lr
import jukemirlib
import numpy as np
import warnings
import argparse
from tqdm import tqdm
import os
from paras import DATA_PATH

warnings.filterwarnings("ignore")

WINDOW_LEN = 20 # limited by jukebox-5B's context length (~23.8s)
LAYER = 53
OUTPUT_DIM = 10 # dimension reduction from 4800 to 10
DOWNSAMPLE_RATE = 100
DOWNSAMPLE_METHOD = 'librosa_kaiser'

def get_window_num(fpath):
    y, sr = lr.load(fpath, sr=None)
    dur = len(y) / sr - 0.01 # 0.01 is the minimum duration for a new window
    return int(dur/WINDOW_LEN) + 1

def ave_pool(jukebox_embedding):
    N = int(jukebox_embedding.shape[1]/OUTPUT_DIM)
    pooled_embedding = []
    for i in range(OUTPUT_DIM):
        emb = jukebox_embedding[:,i*N:i*N+N]
        ave = np.mean(emb, axis=1)
        pooled_embedding.append(ave)
    pooled_embedding = np.stack(pooled_embedding, axis=1)
    return pooled_embedding

def generate_jukebox_embedding(fpath):
    window_num = get_window_num(fpath)
    for i in range(window_num):
        audio = jukemirlib.load_audio(fpath, 
                                      offset=WINDOW_LEN*i, 
                                      duration=WINDOW_LEN)
        acts = jukemirlib.extract(audio, 
                                  downsample_target_rate=DOWNSAMPLE_RATE, 
                                  downsample_method=DOWNSAMPLE_METHOD, 
                                  layers=[LAYER])
        if i == 0:
            full_acts = {num: act for num, act in acts.items()}
        else:
            full_acts = {num: np.concatenate((full_acts[num], act)) \
                         for num, act in acts.items()}
    return full_acts

def compute_for(dataset):
    dataset_dir = os.path.join(DATA_PATH, dataset)
    audio_dir = os.path.join(dataset_dir, f'{dataset}Data')
    juke_dir = os.path.join(dataset_dir, f'{dataset}JukeboxAvePool')
    os.makedirs(juke_dir, exist_ok=True)
    folder_list = []
    subGenreFlag = dataset == 'ballroom' or dataset == 'gtzan'
    if subGenreFlag:
        for genre in os.listdir(audio_dir):
            sub_audio_dir = os.path.join(audio_dir, genre)
            folder_list.append(sub_audio_dir)
    else:
        folder_list.append(audio_dir)
    for audio_folder in folder_list:
        filenames = os.listdir(audio_folder)
        for filename in tqdm(filenames):
            fpath = os.path.join(audio_folder, filename)
            write_file = os.path.splitext(os.path.join(juke_dir, filename))[0]
            if os.path.isfile(write_file+'.npy'):
                continue
            jukebox_embedding = ave_pool(generate_jukebox_embedding(fpath)[LAYER])
            np.save(write_file, jukebox_embedding)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    assert args.dataset in os.listdir(DATA_PATH)
    compute_for(args.dataset)