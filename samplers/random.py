# -*- coding: UTF-8 -*-
from types import SimpleNamespace
from agents.base import BaseLearner
from samplers.base import BaseSampler
import numpy as np


class RandomSampler(BaseSampler):
    """
    RandomIterSampler: A sampler that randomly selects samples from the unlabelled pool.
    """

    def __init__(self, agent: BaseLearner, exp_args: SimpleNamespace, args: SimpleNamespace, ):
        super().__init__(agent, exp_args, args, name='Random')


    def active_learn_task(self, task_stream, i):
        """
        active_learn_task: Selects the next few samples to be labelled randomly.
        """
  
        n_train_samples_per_task = self.get_n_train_samples_per_task()
        n_samples_per_al_cycle = int(n_train_samples_per_task / self.al_total)
        
        task = task_stream.tasks[i]
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = task
        idx_unlabelled = np.arange(x_train.shape[0])
                  
        np.random.shuffle(idx_unlabelled)
        
        n_select = self.al_budget * n_samples_per_al_cycle
        
        # label data by random sampling n_select number of samples
        labelled_idxs = idx_unlabelled[:n_select]
        # update the unlabelled data
        idx_unlabelled = idx_unlabelled[n_select:]

            
        self.agent.learn_task(task, labelled_idxs, new_task=True, args=self.args)

                    
                    