import numpy as np
import librosa
import madmom
import tensorflow as tf
from scipy.ndimage import maximum_filter1d
from tensorflow.keras.utils import Sequence
from scipy.interpolate import interp1d
from scipy.signal import argrelmax
from tqdm import tqdm
from paras import MASK_VALUE, FPS

# infer (global) tempo from beats
def infer_tempo(beats, hist_smooth=15, fps=FPS, no_tempo=MASK_VALUE):
    ibis = np.diff(beats) * fps
    bins = np.bincount(np.round(ibis).astype(int))
    # if no beats are present, there is no tempo
    if not bins.any():
        return no_tempo
    intervals = np.arange(len(bins))
    # smooth histogram bins
    if hist_smooth > 0:
        bins = madmom.audio.signal.smooth(bins, hist_smooth)
    # create interpolation function
    interpolation_fn = interp1d(intervals, bins, 'quadratic')
    # generate new intervals with 1000x the resolution
    intervals = np.arange(intervals[0], intervals[-1], 0.001)
    tempi = 60.0 * fps / intervals
    # apply quadratic interpolation
    bins = interpolation_fn(intervals)
    peaks = argrelmax(bins, mode='wrap')[0]
    if len(peaks) == 0:
        # no peaks, no tempo
        return no_tempo
    else:
        # report only the strongest tempo
        sorted_peaks = peaks[np.argsort(bins[peaks])[::-1]]
        return tempi[sorted_peaks][0]

# pad features
def cnn_pad(data, pad_frames):
    """Pad the data by repeating the first and last frame N times."""
    pad_start = np.repeat(data[:1], pad_frames, axis=0)
    pad_stop = np.repeat(data[-1:], pad_frames, axis=0)
    return np.concatenate((pad_start, data, pad_stop))

# jukebox embedding resampling
def resample(representation, target_rate, method=None):
    if method == None:
        method = 'fft'
    resampled_reps = librosa.resample(np.asfortranarray(representation.T),
                                        FPS, target_rate, res_type=method).T
    return resampled_reps

# wrap training/test data as a Keras sequence
class DataSequence(Sequence):
    def __init__(self, tracks, pre_processor, num_tempo_bins=300, pad_frames=None):
        # store features and targets in dictionaries with name of the song as key
        self.x = {}
        self.beats = {}
        self.downbeats = {}
        self.tempo = {}
        self.ids = []
        self.pad_frames = pad_frames
        # iterate over all tracks
        for i, key in enumerate(tqdm(tracks, desc="wrapping data into Keras sequence")):
            t = tracks[key]
            try:
                # use track only if it contains beats
                beats = t.beats.times
                if t.repr_type == 'audio':
                    # wrap librosa wav data & sample rate as Signal
                    s = madmom.audio.Signal(*t.audio)
                    # compute features first to be able to quantize beats
                    x = pre_processor(s)
                    self.repr_type = 'spec'
                elif t.repr_type == 'jukebox':
                    # resample for augmentation if needed
                    if pre_processor.fps == FPS:
                        x = t.jukebox
                    else:
                        x = resample(t.jukebox, pre_processor.fps)
                    self.repr_type = 'jukebox'
                self.x[key] = x
                # quantize beats
                beats = madmom.utils.quantize_events(beats, fps=pre_processor.fps, length=len(x))
                self.beats[key] = beats
            except AttributeError:
                # no beats found, skip this file
                tqdm.write(f'\r{key} has no beat information, skipping')
                continue
            # downbeats
            try:
                downbeats = t.beats.positions.astype(int) == 1
                downbeats = t.beats.times[downbeats]
                downbeats = madmom.utils.quantize_events(downbeats, fps=pre_processor.fps, length=len(x))
            except AttributeError:
                tqdm.write(f'\r{key} has no downbeat information, masking')
                downbeats = np.ones(len(x), dtype='float32') * MASK_VALUE
            self.downbeats[key] = downbeats
            # tempo
            tempo = None
            try:
                # Note: to be able to augment a dataset, we need to scale the beat times
                tempo = infer_tempo(t.beats.times * pre_processor.fps / 100, fps=pre_processor.fps)
                tempo = tf.keras.utils.to_categorical(int(np.round(tempo)), num_classes=num_tempo_bins, dtype='float32')
            except IndexError:
                # tempo out of bounds (too high)
                tqdm.write(f'\r{key} has no valid tempo ({tempo}), masking')
                tempo = np.ones(num_tempo_bins, dtype='float32') * MASK_VALUE
            self.tempo[key] = tempo
            # keep track of IDs
            self.ids.append(key)
        assert len(self.x) == len(self.beats) == len(self.downbeats) == len(self.tempo) == len(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # convert int idx to key
        if isinstance(idx, int):
            idx = self.ids[idx]
        # Note: we always use a batch size of 1 since the tracks have variable length
        #       keras expects the batch to be the first dimension, the prepend an axis;
        #       append an axis to beats and downbeats as well
        # define features
        x = {}
        data = self.x[idx]
        if self.repr_type == 'spec':
            if self.pad_frames:
                data = cnn_pad(data, self.pad_frames)
            x['spec'] = data[np.newaxis, ..., np.newaxis]
        elif self.repr_type == 'jukebox':
            x['jukebox'] = data[np.newaxis, ...]
        # define targets
        y = {}
        y['beats'] = self.beats[idx][np.newaxis, ..., np.newaxis]
        y['downbeats'] = self.downbeats[idx][np.newaxis, ..., np.newaxis]
        y['tempo'] = self.tempo[idx][np.newaxis, ...]
        return x, y

    def widen_beat_targets(self, size=3, value=0.5):
        for y in self.beats.values():
            # skip masked beat targets
            if np.allclose(y, MASK_VALUE):
                continue
            np.maximum(y, maximum_filter1d(y, size=size) * value, out=y)

    def widen_downbeat_targets(self, size=3, value=0.5):
        for y in self.downbeats.values():
            # skip masked downbeat targets
            if np.allclose(y, MASK_VALUE):
                continue
            np.maximum(y, maximum_filter1d(y, size=size) * value, out=y)

    def widen_tempo_targets(self, size=3, value=0.5):
        for y in self.tempo.values():
            # skip masked tempo targets
            if np.allclose(y, MASK_VALUE):
                continue
            np.maximum(y, maximum_filter1d(y, size=size) * value, out=y)

    def append(self, other):
        assert not any(key in self.ids for key in other.ids), 'IDs must be unique'
        self.x.update(other.x)
        self.beats.update(other.beats)
        self.downbeats.update(other.downbeats)
        self.tempo.update(other.tempo)
        self.ids.extend(other.ids)
