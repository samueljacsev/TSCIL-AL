# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import abc
import numpy as np
import os
from utils.data import Dataloader_from_numpy
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
        print('# Number of samples in the AL cycle:', n_samples_per_al_cycle)
        return n_samples_per_al_cycle #+ 1 # buffer was truncated
    
    def extract_features_and_outputs(self, x_array):

        # Step 1: Extract embeddings for all unlabeled samples
        eval_dataloader = Dataloader_from_numpy(
        x_array,
        np.zeros(len(x_array)),  # Dummy labels
        self.batch_size,
        shuffle=False)

        all_features, all_outputs = [], []
        for batch_id, (batch_x, _) in enumerate(eval_dataloader):
            batch_x = batch_x.to(self.agent.device)
            with torch.no_grad():
                # Extract embeddings and outputs
                features = self.agent.model.feature(batch_x)  # Feature extraction
                outputs = self.agent.model(batch_x)  # Forward pass
            all_features.append(features.cpu().numpy())
            all_outputs.append(outputs)
        all_features = np.vstack(all_features)
        all_outputs = torch.cat(all_outputs, dim=0)

        return all_features, all_outputs



    @abstractmethod
    def active_learn_sampler(self, run, task_stream, task_i):
        """
        active_learn_sample: Abstract method to be implemented by subclasses.
        
        Defines how to select the next batch of samples to be labelled.
        """
        
        NotImplementedError("Subclasses must implement this method.")

    def active_learn_task(self, run, task_stream, task_i):
        """
        active_learn_task: Abstract method to be implemented by subclasses.
        
        Defines how to perform active learning for a given task.
        
        """
        # Set random seeds
        self.random_state = self.args.seed + run
        np.random.seed(self.random_state)  # For NumPy operations
        torch.manual_seed(self.random_state)  # For PyTorch operations
        torch.cuda.manual_seed_all(self.random_state)  # For PyTorch CUDA operations (if using GPU)

        self.current_task = task_stream.tasks[task_i]
        (x_train, y_train) = self.current_task[0]  # y_train is not used for 'unlabeled' data

        n_samples_current_task = x_train.shape[0]

        print('Number of samples in current task:', n_samples_current_task)
        self.n_samples_per_al_cycle = n_samples_current_task // self.al_total

        # Track selection state for downstream samplers
        self.idx_unlabeled = np.arange(n_samples_current_task)
        self.idx_labeled = np.array([], dtype=int)

        print(f'Run: {run}, Task: {task_i}')
        acc_vector = self.active_learn_sampler(run, task_stream, task_i)

        return acc_vector
    
    def save_acc_to_csv(self, accs_data, run, task, cycle, ext=''):
        mean_excd_0 = np.mean(accs_data[accs_data != 0], axis=0)
        print(f'Acc_vector: {accs_data}, Mean: {mean_excd_0} ')
        fn = f'{self.name}{ext}_cycle_{self.al_budget}_per_{self.al_total}_{self.args.data}.csv'
        fn = os.path.join('result',self.args.data , fn)
        save_acc_to_csv(accs_data, run, task, cycle, filename=fn)
    


