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

"""A base class for managing setting up experiments."""

from abc import ABCMeta, abstractmethod

class ExperimentBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train_once(iteration, presets=None, masks=None):
        """
        Trains a model
        Args:
            iteration: the current iteration of the experiment
            presets: a dict of initial weights of the network
            masks: a dict of masks for the weights of the network

        Return: the model's initial and final weights as dictionaries, as well as the final train acc.
        """
        raise NotImplementedError

    @abstractmethod
    def prune_masks(masks, final_weights):
        """
        Args:
            masks: dictionary of current masks
            final_weights: dictionary of final weights
        Returns a new dictionary of masks that have been pruned. Each dictionary
        key is the name of a tensor in the network; each value is a numpy array
        containing the values of the tensor (1/0 values for mask, weights for the dictionary of final weights).
        """
        raise NotImplementedError

    @abstractmethod
    def stop_iterating(train_acc):
        """
        Should the experiment stop iterating before the total number of iterations is finished?
        Args:
            train_acc: the final train accuracy of the last iteration
        """
        raise NotImplementedError
