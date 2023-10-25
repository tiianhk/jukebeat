import os
import argparse
import tensorflow as tf
import numpy as np
import random
from paras import TRAIN_DATASET, RANDOM_SEED
from load_data import get_tracks
from train import Trainer
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Jukebeat')
    parser.add_argument('--gpu', required=True)
    parser.add_argument('-d', '--datasets', nargs='+', default=TRAIN_DATASET)
    parser.add_argument('-f', '--fold', type=int, required=True)
    parser.add_argument('--input_repr', choices=['audio', 'jukebox'], required=True)
    parser.add_argument('--augmentation', action='store_true')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    # Configure GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Load data & train models
    tracks = get_tracks(args.datasets, args.input_repr)
    dataset_used = '-'.join(args.datasets)
    augmented = 'aug' if args.augmentation else 'not_aug'
    output_dir = f'{args.input_repr}_{augmented}_{dataset_used}/fold{args.fold}'
    trainer = Trainer(tracks, args.fold, output_dir, 
                      args.datasets, augmentation=args.augmentation)
    trainer.train()
    trainer.evaluate_on_each_dataset()
