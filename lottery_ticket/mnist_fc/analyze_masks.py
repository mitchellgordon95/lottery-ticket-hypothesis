from __future__ import print_function

import os
import sys
import subprocess
import numpy as np
import re
from lottery_ticket.foundations import paths
from lottery_ticket.mnist_fc import constants

class AveragePrinter(object):
    def __init__(self, do_avg):
        self.do_avg = do_avg
        self.counters = {}
        self.print_order = []

    def do_print(self, trial, text, format_vals):
        """Note: format vals must be numeric"""
        if text not in self.print_order:
            self.print_order.append(text)

        self.counters[text] = self.counters.get(text, [])
        self.counters[text].append(format_vals)

    def flush(self):
        if self.do_avg:
            print('Trial Averages')
            for text in self.print_order:
                counters = list(np.array(self.counters[text]).mean(axis=0))
                print(text.format(*counters))
        else:
            for trial in range(len(self.counters[self.print_order[0]])):
                print('Trial {}'.format(trial + 1))
                for text in self.print_order:
                    print(text.format(*self.counters[text][trial]))


avg_printer = AveragePrinter(len(sys.argv) > 2 and sys.argv[2] == '-a')
exp_path = paths.experiment(constants.EXPERIMENT_PATH, sys.argv[1])
trial_nums = [int(re.findall('\d+', trial_dir)[0]) for trial_dir in os.listdir(exp_path)]
print("Found {} trials".format(max(trial_nums)))
for trial in range(1, max(trial_nums) + 1):
    trial_path = paths.trial(exp_path, trial)
    if not os.path.isdir(trial_path):
        print("Warning: skipping trial {}, does not exist".format(trial))
        continue

    first_run_path = paths.run(trial_path, 0)
    first_run_train_acc = float(subprocess.check_output(['tail', '-n', '1', paths.log(first_run_path, 'train')]).strip().split(',')[-1])
    avg_printer.do_print(trial, '\tFirst run train acc: {}', [first_run_train_acc])
    first_run_test_acc = float(subprocess.check_output(['tail', '-n', '1', paths.log(first_run_path, 'test')]).strip().split(',')[-1])
    avg_printer.do_print(trial, '\tFirst run test acc: {}', [first_run_test_acc])

    runs = map(int, os.listdir(trial_path))
    second_last_run = sorted(runs)[-2] if len(runs) > 1 else runs[0]
    second_last_path = paths.run(trial_path, second_last_run)

    second_last_train_acc = float(subprocess.check_output(['tail', '-n', '1', paths.log(second_last_path, 'train')]).strip().split(',')[-1])
    avg_printer.do_print(trial, '\tSecond to last train acc run ({}): {}', [second_last_run, second_last_train_acc])
    second_last_test_acc = float(subprocess.check_output(['tail', '-n', '1', paths.log(second_last_path, 'test')]).strip().split(',')[-1])
    avg_printer.do_print(trial, '\tTest: {}', [second_last_test_acc])
    avg_printer.do_print(trial, '\t{:^20s}{:^20s}{:^20s}{:^20s}'.format('Nonempty', 'Rows', 'Columns', 'Weights'), ())
    if second_last_run == 0:
        # First runs don't have masks, so we have to use the initial weights to get the shapes of the layers
        weights_dir = paths.initial(second_last_path)
        for mask_name in sorted(os.listdir(weights_dir)):
            mask = np.load(os.path.join(weights_dir, mask_name))
            avg_printer.do_print(trial, '\t{:^20s}'.format(mask_name) + '{:>10.2f}/{:<10.2f}{:>10.2f}/{:<10.2f}{:>10.2f}/{:<10.2f}', [mask.shape[0], mask.shape[0],
                                                                            mask.shape[1], mask.shape[1],
                                                                            mask.shape[0] * mask.shape[1], mask.shape[0] * mask.shape[1]])
    else:
        masks_dir = paths.masks(second_last_path)
        for mask_name in sorted(os.listdir(masks_dir)):
            mask = np.load(os.path.join(masks_dir, mask_name))
            avg_printer.do_print(trial, '\t{:^20s}'.format(mask_name) + '{:>10.2f}/{:<10.2f}{:>10.2f}/{:<10.2f}{:>10.2f}/{:<10.2f}', [np.sum(np.sum(mask,axis=1) > 0), mask.shape[0],
                                                                            np.sum(np.sum(mask,axis=0) > 0), mask.shape[1],
                                                                            np.sum(mask), mask.shape[0] * mask.shape[1]])

avg_printer.flush()
