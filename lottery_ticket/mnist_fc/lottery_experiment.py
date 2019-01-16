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

"""Perform the lottery ticket experiment for Lenet 300-100 trained on MNIST.

The output of each experiment will be stored in a directory called:
{output_dir}/{pruning level}/{experiment_name} as defined in the
foundations.paths module.

Args:
  output_dir: Parent directory for all output files.
  mnist_location: The path to the NPZ file containing MNIST.
  training_len: How long to train on each iteration.
  iterations: How many iterative pruning steps to perform.
  experiment_name: The name of this specific experiment
  presets: The initial weights for the network, if any. Presets can come in
    one of three forms:
    * A dictionary of numpy arrays. Each dictionary key is the name of the
      corresponding tensor that is to be initialized. Each value is a numpy
      array containing the initializations.
    * The string name of a directory containing one file for each
      set of weights that is to be initialized (in the form of
      foundations.save_restore).
    * None, meaning the network should be randomly initialized.
  permute_labels: Whether to permute the labels on the dataset.
  train_order_seed: The random seed, if any, to be used to determine the
    order in which training examples are shuffled before being presented
    to the network.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import fire
import tensorflow as tf
from lottery_ticket.datasets import dataset_mnist
from lottery_ticket.foundations import experiment
from lottery_ticket.foundations import model_fc
from lottery_ticket.foundations import paths
from lottery_ticket.foundations import pruning
from lottery_ticket.foundations import save_restore
from lottery_ticket.foundations import trainer
from lottery_ticket.foundations.experiment_base import ExperimentBase
from lottery_ticket.mnist_fc import constants

class MnistExperiment(ExperimentBase):
  def __init__(
      self,
      output_dir,
      mnist_location,
      permute_labels,
      train_order_seed,
      training_len,
      experiment_name):
    self.output_dir = output_dir
    self.mnist_location = mnist_location
    self.permute_labels = permute_labels
    self.train_order_seed = train_order_seed
    self.training_len = training_len
    self.experiment_name = experiment_name

  def train_once(self, iteration, presets=None, masks=None):
    tf.reset_default_graph()
    sess = tf.Session()
    dataset = dataset_mnist.DatasetMnist(
        self.mnist_location,
        permute_labels=self.permute_labels,
        train_order_seed=self.train_order_seed)
    input_tensor, label_tensor = dataset.placeholders
    model = model_fc.ModelFc(constants.HYPERPARAMETERS, input_tensor, label_tensor, presets=presets, masks=masks)
    params = {
        'test_interval': 100,
        'save_summaries': True,
        'save_network': True,
    }
    return trainer.train(
        sess,
        dataset,
        model,
        constants.OPTIMIZER_FN,
        self.training_len,
        output_dir=paths.run(self.output_dir, iteration, self.experiment_name),
        **params)

  def prune_masks(self, masks, final_weights):
    return pruning.prune_by_percent(constants.PRUNE_PERCENTS, masks, final_weights)

  def stop_iterating(self, final_acc):
    return False


def main(
    trials=20
    mnist_location=constants.MNIST_LOCATION,
    training_len=constants.TRAINING_LEN,
    iterations=30,
    experiment_name='same_init',
    presets=None,
    permute_labels=False,
    train_order_seed=None):
  for trial in range(1, trials+1):
    mnist_experiment = MnistExperiment(
        output_dir=constants.trial(trial),
        mnist_location,
        permute_labels=permute_labels,
        train_order_seed=train_order_seed,
        training_len=training_len,
        experiment_name=experiment_name)

    experiment.run_experiment(
        mnist_experiment,
        iterations,
        presets=save_restore.standardize(presets))

if __name__ == '__main__':
  fire.Fire(main)
