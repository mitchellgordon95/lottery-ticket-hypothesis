from __future__ import print_function

import os
import sys
import subprocess
import numpy as np
from lottery_ticket.foundations import paths
from lottery_ticket.mnist_fc import constants


for trial in range(1, 21):
    print('Trial {}'.format(trial))
    trial_path = paths.trial(constants.EXPERIMENT_PATH, trial, sys.argv[1])

    first_run_path = paths.run(trial_path, 0)
    first_run_train_acc = subprocess.check_output(['tail', '-n', '1', paths.log(first_run_path, 'train')]).strip().split(',')[-1]
    print('\tFirst run train acc: {}'.format(first_run_train_acc))
    first_run_test_acc = subprocess.check_output(['tail', '-n', '1', paths.log(first_run_path, 'test')]).strip().split(',')[-1]
    print('\tFirst run test acc: {}'.format(first_run_test_acc))

    runs = map(int, os.listdir(trial_path))
    second_last_run = sorted(runs)[-2]
    second_last_path = paths.run(trial_path, second_last_run)

    second_last_train_acc = subprocess.check_output(['tail', '-n', '1', paths.log(second_last_path, 'train')]).strip().split(',')[-1]
    print('\tSecond to last train acc run ({}): {}'.format(second_last_run, second_last_train_acc))
    second_last_test_acc = subprocess.check_output(['tail', '-n', '1', paths.log(second_last_path, 'test')]).strip().split(',')[-1]
    print('\tTest: {}'.format(second_last_test_acc))
    print('\tNonempty\tRows\tColumns\tWeights')
    masks_dir = paths.masks(second_last_path)
    for mask in sorted(os.listdir(masks_dir)):
        print('\t' + mask, end='')
        mask = np.load(os.path.join(masks_dir, mask))
        print('\t{}/{}'.format(np.sum(np.sum(mask,axis=1) > 0), mask.shape[0]), end='')
        print('\t{}/{}'.format(np.sum(np.sum(mask,axis=0) > 0), mask.shape[1]), end='')
        print('\t{}/{}'.format(np.sum(mask), mask.shape[0] * mask.shape[1]))
