# Copyright (C) 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the lottery ticket experiment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def run_experiment(experiment, max_prune_iterations, presets=None):
  """Run the lottery ticket experiment for the specified number of iterations.

  Args:
    experiment: an object implementing ExperimentBase
    max_num_prunes: The number of pruning iterations to perform.
    presets: (optional) The presets to use for the first iteration of training.
      In the form of a dictionary where each key is the name of a tensor and
      each value is a numpy array of the values to which that tensor should
      be initialized.
  """
  # Run once normally.
  initial, final_weights, train_acc = experiment.train_once(0, presets=presets)

  # Create the initial masks with no weights pruned.
  masks = {}
  for k, v in initial.items():
    masks[k] = np.ones(v.shape)

  # Begin the training loop.
  for iteration in range(1, max_prune_iterations + 1):
    if experiment.stop_pruning(train_acc):
      break

    # Prune the network.
    masks = experiment.prune_masks(masks, final_weights)

    # Train the network again.
    _, final_weights, train_acc = experiment.train_once(iteration, presets=initial, masks=masks)
