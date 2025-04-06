# -*- coding: UTF-8 -*-
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import abc
import os
from abc import abstractmethod
from types import SimpleNamespace
from agents.base import BaseLearner
from utils.setup_elements import get_buffer_size, n_smp_per_cls, n_classes, n_tasks





class BaseSampler(nn.Module, metaclass=abc.ABCMeta):
    """
    Base class for Samplers.
    """

    def __init__(self, 
                 agent: BaseLearner, 
                 exp_args:SimpleNamespace, 
                 args:SimpleNamespace, 
                 name: str='base'):
        super(BaseSampler, self).__init__()
        self.name = name
        self.exp_args = exp_args
        self.args = args
        self.agent = agent
        self.batch_size = agent.batch_size
        self.al_budget = exp_args.al_budget
        self.al_total = exp_args.al_total
        self.buffer_size = get_buffer_size(args)
        
        print('AL strategy:', self.name)

    def get_n_train_samples_per_task(self):
        train_ratio = 0.8
        n_samples_per_class = n_smp_per_cls[self.args.data]
        n_classes_per_task = n_classes[self.args.data] / n_tasks[self.args.data] # 2 by default
        
        return int(train_ratio * n_samples_per_class * n_classes_per_task)
    

    @abstractmethod
    def active_learn_task(self):
        """
        active_learn_task: Abstract method to be implemented by subclasses.
        
        Defines how to select the next batch of samples to be labelled.
        
        """
        raise NotImplementedError("This method should be overridden by subclasses.")