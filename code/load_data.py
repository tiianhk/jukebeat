import os
import re
import numpy as np
import librosa
from sklearn.model_selection import KFold
from paras import DATA_PATH, RANDOM_SEED

""" load data like mirdata """

class beats_annotation_loader():
    def __init__(self, beats_path):
        times = []
        positions = []
        with open(beats_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = re.split(' |\t', line.rstrip())
            times.append(float(line[0]))
            # smc only have beat annotations
            if len(line) > 1:
                positions.append(float(line[1]))
        self.times = np.array(times)
        if len(positions) > 0:
            self.positions = np.array(positions)
            assert len(self.times) == len(self.positions)

class track_loader():
    def __init__(self, audio_path, beats_path, jukebox_path, genre, dataset, repr_type):
        self.audio_path = audio_path
        self.beats_path = beats_path
        self.jukebox_path = jukebox_path
        self.genre = genre
        self.dataset = dataset
        self.repr_type = repr_type
        # load representations
        if repr_type == 'audio':
            self.audio = librosa.load(audio_path, sr=None)
        elif repr_type == 'jukebox':
            self.jukebox = np.load(jukebox_path)
        # load annotations
        self.beats = beats_annotation_loader(beats_path)

def get_tracks(datasets, repr_type):
    tracks = {}
    # for 8-fold cross validation
    kf = KFold(n_splits=8, shuffle=True, random_state=RANDOM_SEED)
    for dataset in datasets:
        print(f'loading tracks from {dataset}')
        AUDIO_PATH = os.path.join(DATA_PATH, f'{dataset}/{dataset}Data/')
        BEATS_PATH = os.path.join(DATA_PATH, f'{dataset}/{dataset}Annotations/')
        JUKEBOX_PATH = os.path.join(DATA_PATH, f'{dataset}/{dataset}JukeboxAvePool/')
        folder_list = []
        # ballroom and gtzan have sub-genres
        subGenreFlag = dataset == 'ballroom' or dataset == 'gtzan'
        # add audio folders to a folder list
        if subGenreFlag:
            for genre in os.listdir(AUDIO_PATH):
                audio_dir = os.path.join(AUDIO_PATH, genre)
                folder_list.append(audio_dir)
        else:
            folder_list.append(AUDIO_PATH)
        # load data
        key_list = []
        for audio_folder in folder_list:
            filenames = os.listdir(audio_folder)
            for filename in filenames:
                filename_without_ext = os.path.splitext(filename)[0]
                audiofile_path = os.path.join(audio_folder, filename)
                beatsfile_path = os.path.join(BEATS_PATH, filename_without_ext+'.beats')
                """
                    ballroom (audio dataset) has duplicated pieces.
                    annotations of those are not provided.
                    see https://github.com/CPJKU/BallroomAnnotations/blob/master/README.md
                """
                if not os.path.isfile(beatsfile_path):
                    print(f'{beatsfile_path} missing, skipped..')
                    continue
                jukeboxfile_path = os.path.join(JUKEBOX_PATH, filename_without_ext+'.npy')
                genre = os.path.basename(audio_folder) if subGenreFlag else None
                tracks[filename_without_ext] = track_loader(audiofile_path, beatsfile_path, 
                                                            jukeboxfile_path, genre, 
                                                            dataset, repr_type)
                key_list.append(filename_without_ext)
        # generate fold id
        for i, (_, test_index) in enumerate(kf.split(key_list)):
            for idx in test_index:
                tracks[key_list[idx]].fold_id = i
    return tracks
    