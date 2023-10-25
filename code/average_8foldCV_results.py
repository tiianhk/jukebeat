import os
import re
import sys
import numpy as np
from paras import TRAIN_DATASET, DATA_PATH

if __name__ == '__main__':

    exp_dir = os.path.join(f'{DATA_PATH}/../exp', sys.argv[1])
    for dataset in TRAIN_DATASET:
        beat1 = []
        beat2 = []
        downbeat1 = []
        downbeat2 = []
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
                        beat1.append(float(line[1]))
                    elif i == 5:
                        beat2.append(float(line[1]))
                    elif i == 9:
                        downbeat1.append(float(line[1]))
                    elif i == 11:
                        downbeat2.append(float(line[1]))
        print(f'{dataset}:')
        print(f'beat: {np.mean(beat1):.3f}, {np.mean(beat2):.3f}')
        if dataset != 'smc':
            print(f'downbeat: {np.mean(downbeat1):.3f}, {np.mean(downbeat2):.3f}')