# -*- coding: UTF-8 -*-
import torch.nn as nn
import abc
import numpy as np
import os
from abc import abstractmethod
from types import SimpleNamespace
from agents.base import BaseLearner
from utils.metrics import compute_performance
from result.utils import save_acc_to_csv


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
        return n_samples_per_al_cycle + 1 # buffer was truncated
    
    def save_acc_to_csv(self, accs_data, run, task, cycle, ext=''):
        fn = f'{self.name}{ext}_cycle_{self.al_budget}_{self.args.data}.csv'
        fn = os.path.join('result',self.args.data , fn)
        save_acc_to_csv(accs_data, run, task, cycle, filename=fn)
    
    
    def calculate_metrcs(self, acc_matrix):
        acc_matrix = [np.array(acc_matrix)]
        acc_matrix = np.array(acc_matrix)
        
        avg_end_acc, avg_end_fgt, avg_cur_acc, avg_acc = compute_performance(acc_matrix)
        print('Average end accuracy:', avg_end_acc)
        print('Average end forgetting:', avg_end_fgt)
        print('Average current accuracy:', avg_cur_acc)
        print('Average accuracy:', avg_acc)
        #print('Average BWT+:', avg_bwtp)
        


    @abstractmethod
    def active_learn_task(self, run, task_stream, task_i):
        """
        active_learn_task: Abstract method to be implemented by subclasses.
        
        Defines how to select the next batch of samples to be labelled.
        
        """
        raise NotImplementedError("This method should be overridden by subclasses.")