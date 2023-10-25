import os
import numpy as np
import madmom
from scipy.ndimage import maximum_filter1d
from scipy.interpolate import interp1d
from scipy.signal import argrelmax
from postprocess import beat_tracker, downbeat_tracker, bar_tracker
from paras import FPS

def detect_tempo(bins, hist_smooth=11, min_bpm=10):
    min_bpm = int(np.floor(min_bpm))
    tempi = np.arange(min_bpm, len(bins))
    bins = bins[min_bpm:]
    # smooth histogram bins
    if hist_smooth > 0:
        bins = madmom.audio.signal.smooth(bins, hist_smooth)
    # create interpolation function
    interpolation_fn = interp1d(tempi, bins, 'quadratic')
    # generate new intervals with 1000x the resolution
    tempi = np.arange(tempi[0], tempi[-1], 0.001)
    # apply quadratic interpolation
    bins = interpolation_fn(tempi)
    peaks = argrelmax(bins, mode='wrap')[0]
    if len(peaks) == 0:
        # no peaks, no tempo
        tempi = np.array([], ndmin=2)
    elif len(peaks) == 1:
        # report only the strongest tempo
        ret = np.array([tempi[peaks[0]], 1.0])
        tempi = np.array([tempi[peaks[0]], 1.0])
    else:
        # sort the peaks in descending order of bin heights
        sorted_peaks = peaks[np.argsort(bins[peaks])[::-1]]
        # normalize their strengths
        strengths = bins[sorted_peaks]
        strengths /= np.sum(strengths)
        # return the tempi and their normalized strengths
        ret = np.array(list(zip(tempi[sorted_peaks], strengths)))
        tempi = np.array(list(zip(tempi[sorted_peaks], strengths)))
    return tempi[:2]

# function to predict the model's output(s), post-process everything and save if needed
def predict(model, dataset, detdir=None):
    activations = {}
    detections = {}
    for i, t in enumerate(dataset):
        if isinstance(dataset, list):
            x = t[0]
            f = t[1]
        else:
            x = t[0]
            f = dataset.ids[i]
        # predict activations
        beats, downbeats, tempo = model.predict(x, verbose=0)
        beats_act = beats.squeeze()
        downbeats_act = downbeats.squeeze()
        tempo_act = tempo.squeeze()
        # beats
        beats = beat_tracker(beats_act)
        # downbeats
        combined_act = np.vstack((np.maximum(beats_act - downbeats_act, 0), downbeats_act)).T
        downbeats = downbeat_tracker(combined_act)
        # bars (i.e. track beats and then downbeats)
        beat_idx = (beats * FPS).astype(np.int_)
        bar_act = maximum_filter1d(downbeats_act, size=3)
        bar_act = bar_act[beat_idx]
        bar_act = np.vstack((beats, bar_act)).T
        try:
            bars = bar_tracker(bar_act)
        except IndexError:
            bars = np.empty((0, 2))
        # tempo
        tempo = detect_tempo(tempo_act)
        # collect activations and detections
        activations[f] = {'beats': beats_act, 'downbeats': downbeats_act, 'combined': combined_act, 'tempo': tempo_act}
        detections[f] = {'beats': beats, 'downbeats': downbeats, 'bars': bars, 'tempo': tempo}
        # save activations & detections
        if detdir is not None:
            os.makedirs(detdir, exist_ok=True)
            np.save('%s/%s.beats.npy' % (detdir, f), beats_act)
            np.save('%s/%s.downbeats.npy' % (detdir, f), downbeats_act)
            np.save('%s/%s.tempo.npy' % (detdir, f), tempo_act)
            madmom.io.write_beats(beats, '%s/%s.beats.txt' % (detdir, f))
            madmom.io.write_beats(downbeats, '%s/%s.downbeats.txt' % (detdir, f))
            madmom.io.write_beats(bars, '%s/%s.bars.txt' % (detdir, f))
            madmom.io.write_tempo(tempo, '%s/%s.bpm.txt' % (detdir, f))
    return activations, detections

def evaluate_beats(detections, annotations):
    evals = []
    for key, det in detections.items():
        ann = annotations[key]
        e = madmom.evaluation.beats.BeatEvaluation(det, ann)
        evals.append(e)
    return madmom.evaluation.beats.BeatMeanEvaluation(evals)

def evaluate_downbeats(detections, annotations):
    evals = []
    for key, det in detections.items():
        try:
            ann = annotations[key]
        except KeyError:
            print(f'{key} not found in annotations, skipped')
            continue
        e = madmom.evaluation.beats.BeatEvaluation(det, ann, downbeats=True)
        evals.append(e)
    return madmom.evaluation.beats.BeatMeanEvaluation(evals)

def evaluate_tempo(detections, annotations):
    evals = []
    for key, det in detections.items():
        ann = annotations[key]
        e = madmom.evaluation.tempo.TempoEvaluation(det, ann)
        evals.append(e)
    return madmom.evaluation.tempo.TempoMeanEvaluation(evals)
