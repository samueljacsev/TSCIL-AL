# -*- coding: UTF-8 -*-
from types import SimpleNamespace
from agents.base import BaseLearner
from samplers.base import BaseSampler
import numpy as np


class RandomSampler(BaseSampler):
    """
    RandomIterSampler: A sampler that randomly selects samples from the unlabelled pool.
    """

    def __init__(self, 
                 agent: BaseLearner,
                 exp_args: SimpleNamespace,
                 args: SimpleNamespace ):
        super().__init__(agent, exp_args, args, name='Random')


    def active_learn_task(self, run, task_stream, task_i):
        """
        active_learn_task: Selects the next few samples to be labelled randomly.
        """
        
        task = task_stream.tasks[task_i]
        x_train = task[0][0]
        n_samples_per_al_cycle = self.get_n_samples_per_al_cycle(x_train.shape[0])
        
        idx_unlabelled = np.arange(x_train.shape[0])
        np.random.shuffle(idx_unlabelled)
        
        n_select = self.al_budget * n_samples_per_al_cycle
        labelled_idxs = idx_unlabelled[:n_select]

        
        self.agent.learn_task(task, labelled_idxs)
        print('ln 36')
        alc = 0 # the only active learning cycle
        accuracies = self.agent.evaluate(task_stream, alc, self.al_budget)
        self.save_acc_to_csv(accuracies, run, task_i, alc)
        
            