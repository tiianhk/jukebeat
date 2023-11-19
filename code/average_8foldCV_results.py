import os
import re
import sys
import numpy as np
from paras import TRAIN_DATASET, DATA_PATH

if __name__ == '__main__':

    exp_dir = os.path.join(f'{DATA_PATH}/../exp', sys.argv[1])
    for dataset in TRAIN_DATASET:
        beat_fmeasure, beat_cmlt, beat_amlt = [],[],[]
        downbeat_fmeasure, downbeat_cmlt, downbeat_amlt = [],[],[]
        for k in range(8):
            fold_path = os.path.join(exp_dir, f'fold{k}')
            if not os.path.isdir(fold_path):
                continue
            eval_file_path = os.path.join(fold_path, f'{dataset}-eval.txt')
            with open(eval_file_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    line = re.split(' ', line.strip())
                    if i == 3:
                        beat_fmeasure.append(float(line[1]))
                        beat_cmlt.append(float(line[11]))
                        beat_amlt.append(float(line[15]))
                    elif i == 11:
                        downbeat_fmeasure.append(float(line[1]))
                        downbeat_cmlt.append(float(line[11]))
                        downbeat_amlt.append(float(line[15]))
        print(f'{dataset}:')
        print(f'beat: f_measure {np.mean(beat_fmeasure):.3f} CMLt {np.mean(beat_cmlt):.3f} AMLt {np.mean(beat_amlt):.3f}')
        if dataset != 'smc':
            print(f'downbeat: f_measure {np.mean(downbeat_fmeasure):.3f} CMLt {np.mean(downbeat_cmlt):.3f} AMLt {np.mean(downbeat_amlt):.3f}')