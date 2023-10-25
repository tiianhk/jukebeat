import madmom
from paras import FPS

# track beats with a DBN
beat_tracker = madmom.features.beats.DBNBeatTrackingProcessor(
    min_bpm=55.0, max_bpm=215.0, fps=FPS, transition_lambda=100, threshold=0.05
)

# track downbeats with a DBN
# as input, use a combined beat & downbeat activation function
downbeat_tracker = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
    beats_per_bar=[3, 4], min_bpm=55.0, max_bpm=215.0, fps=FPS, transition_lambda=100
)

# track bars, i.e. first track the beats and then infer the downbeat positions
bar_tracker = madmom.features.downbeats.DBNBarTrackingProcessor(
    beats_per_bar=(3, 4), meter_change_prob=1e-3, observation_weight=4
)