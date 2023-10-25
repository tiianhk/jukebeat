import tensorflow as tf
import tensorflow_addons as tfa
import os
import sys
import tensorflow.keras.backend as K
from preprocess import PreProcessor
from data_sequence import DataSequence
from model import create_model
from load_data import get_tracks
from evaluate import predict, evaluate_beats, evaluate_downbeats
from paras import MASK_VALUE, MAX_EPOCH, LEARNING_RATE, CLIP_NORM, DATA_PATH

def build_masked_loss(loss_function, mask_value=MASK_VALUE):
    """Builds a loss function that masks based on targets

    Args:
        loss_function: The loss function to mask
        mask_value: The value to mask in the targets

    Returns:
        function: a loss function that acts like loss_function with masked inputs
    """
    def masked_loss_function(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        return loss_function(y_true * mask, y_pred * mask)
    return masked_loss_function

"""
    the tutorial didn't use it
    should be used as metrics in model.compile (?)
"""
def masked_accuracy(y_true, y_pred):
    total = K.sum(K.not_equal(y_true, MASK_VALUE))
    correct = K.sum(K.equal(y_true, K.round(y_pred)))
    return correct / total

class Trainer():
    def __init__(self, tracks, fold_id, 
                 output_dir, datasets, 
                 augmentation=True, 
                 epochs=MAX_EPOCH, 
                 learnrate=LEARNING_RATE,
                 clipnorm=CLIP_NORM):
        self.outdir = os.path.join(DATA_PATH, f'../exp/{output_dir}')
        os.makedirs(self.outdir, exist_ok=True)
        self.tracks = tracks
        self._wrap_data_into_keras_sequence(fold_id, augmentation)
        self.epochs = epochs
        self.learnrate = learnrate
        self.clipnorm = clipnorm
        self.datasets = datasets
        self._log_output()
    
    def _log_output(self):
        logfile = os.path.join(self.outdir, 'output.log')
        output_stream = open(logfile, "w")
        sys.stdout = output_stream
        sys.stderr = output_stream

    def _get_train_test_files(self, fold_id):
        train_files, test_files = [], []
        for file, t in self.tracks.items():
            if t.fold_id == fold_id:
                test_files.append(file)
            else:
                train_files.append(file)
        return train_files, test_files

    def _wrap_data_into_keras_sequence(self, fold_id, augmentation):
        train_files, test_files = self._get_train_test_files(fold_id)
        print('preparing train set')
        self.train_split = self._get_ds(train_files)
        if augmentation:
            for fps in [95, 97.5, 102.5, 105]:
                print(f'augmenting data with fps={fps}')
                self.train_split.append(self._get_ds(train_files, fps=fps))
        print('preparing test data')
        self.test_split = self._get_ds(test_files)

    def _get_ds(self, files, fps=100, pad_frames=2):
        ds = DataSequence(
            tracks={k if fps==100 else f'{k}_{fps}': v for k, v in self.tracks.items() if k in files}, 
            pre_processor=PreProcessor(fps=fps), pad_frames=pad_frames
        )
        ds.widen_beat_targets()
        ds.widen_downbeat_targets()
        ds.widen_tempo_targets()
        ds.widen_tempo_targets()
        return ds

    def train(self):
        repr_type = self.train_split.repr_type
        example = self.train_split[0][0][repr_type]
        if repr_type == 'spec':
            input_shape = (None,) + example.shape[-2:]
        elif repr_type == 'jukebox':
            input_shape = (None,) + example.shape[-1:]
        model = create_model(input_shape, repr_type)
        optimizer = tfa.optimizers.Lookahead(
                    tfa.optimizers.RectifiedAdam(learning_rate=self.learnrate, 
                                                 clipnorm=self.clipnorm), 
                    sync_period=5)
        model.compile(
            optimizer=optimizer,
            loss=[
                build_masked_loss(K.binary_crossentropy),
                build_masked_loss(K.binary_crossentropy),
                build_masked_loss(K.binary_crossentropy),
            ],
            metrics=['binary_accuracy'],
        )
        # model checkpointing
        mc = tf.keras.callbacks.ModelCheckpoint(f'{self.outdir}/model_best.h5',
                                                monitor='loss', save_best_only=True,
                                                verbose=0)
        # learn rate scheduler
        lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                                                  patience=10, verbose=2, mode='auto',
                                                  min_delta=1e-3, cooldown=0, min_lr=1e-7)
        # early stopping
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-4,
                                              patience=20, verbose=0)
        # tensorboard logging
        tb = tf.keras.callbacks.TensorBoard(log_dir=f'{self.outdir}/logs',
                                            write_graph=True, write_images=True)
        # actually train network
        # use test split as validation data only to monitor the progress
        model.fit(
            x = self.train_split,
            epochs = self.epochs,
            verbose = 2,
            callbacks=[mc, es, tb, lr],
            validation_data=self.test_split,
            shuffle=True,
            steps_per_epoch=len(self.train_split),
            validation_steps=len(self.test_split),
        )
        model.save(f'{self.outdir}/model_final.h5')

    def _fetch_test_split_of_single_dataset(self, dataset):
        data = []
        for i, item in enumerate(self.test_split):
            file = self.test_split.ids[i]
            if self.tracks[file].dataset == dataset:
                data.append((item[0], file))
        return data

    def evaluate_on_each_dataset(self):
        # load model
        model = tf.keras.models.load_model(f'{self.outdir}/model_final.h5', compile=False)
        # evaluate on the test split of each dataset
        for dataset in self.datasets:
            print(f'evaluating on test split of {dataset}')
            # fetch data from the combined test split
            data = self._fetch_test_split_of_single_dataset(dataset)
            # model inference & post-processing
            _, detections = predict(model, data)
            beat_detections = {k: v['beats'] for k, v in detections.items()}
            downbeat_detections = {k: v['downbeats'] for k, v in detections.items()}
            bar_detections = {k: v['bars'] for k, v in detections.items()}
            test_files = [item[1] for item in data]
            beat_annotations = {k: v.beats.times for k, v in self.tracks.items() if k in test_files}
            ifDownbeatAnnotated = 1
            try:
                downbeat_annotations = {k: v.beats.times[v.beats.positions == 1] for k, v in self.tracks.items() if k in test_files}
            except AttributeError:
                ifDownbeatAnnotated = 0
            # output evaluations to a file
            resdir = f'{self.outdir}/{dataset}-eval.txt'
            with open(resdir,'a') as f:
                f.write('Beat evaluation\n---------------\n Beat tracker:    ')
                f.write(evaluate_beats(beat_detections, beat_annotations).tostring())
                f.write('\n Downbeat tracker:    ')
                f.write(evaluate_beats(downbeat_detections, beat_annotations).tostring())
                if ifDownbeatAnnotated:
                    f.write('\nDownbeat evaluation\n-------------------\n  Bar tracker:     ')
                    f.write(evaluate_downbeats(bar_detections, downbeat_annotations).tostring())
                    f.write('\n Downbeat tracker:    ')
                    f.write(evaluate_downbeats(downbeat_detections, downbeat_annotations).tostring())
                f.write('\n')
