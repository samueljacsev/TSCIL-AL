# -*- coding: UTF-8 -*-
import torch.nn as nn
import abc
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
        super().__init__()
        
        self.name = name
        self.args = args
        self.agent = agent
        self.batch_size = agent.batch_size
        self.al_budget = exp_args.al_budget
        self.al_total = exp_args.al_total
        
        print('AL strategy:', self.name)
        print('Batch size:', self.batch_size)
        

    def get_n_samples_per_al_cycle(self, n_samples_current_task):
        """
        get_n_of_al_cycles: Calculate the number of active learning cycles.
        
        Returns:
            Number of active learning cycles.
        """

        # Number of samples to label per AL cycle 
        n_samples_per_al_cycle = n_samples_current_task // self.al_total
        print('Number of samples per AL cycle:', n_samples_per_al_cycle)
        return n_samples_per_al_cycle
        


    @abstractmethod
    def active_learn_task(self, task_stream, i):
        """
        active_learn_task: Abstract method to be implemented by subclasses.
        
        Defines how to select the next batch of samples to be labelled.
        
        """
        raise NotImplementedError("This method should be overridden by subclasses.")