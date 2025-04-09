# -*- coding: UTF-8 -*-
from types import SimpleNamespace
from agents.base import BaseLearner
from samplers.base import BaseSampler
import numpy as np


class RandomIterSampler(BaseSampler):
    """
    RandomIterSampler: A sampler that randomly selects samples from the unlabelled pool.
    """

    def __init__(self, 
                 agent: BaseLearner,
                 exp_args: SimpleNamespace,
                 args: SimpleNamespace ):
        super().__init__(agent, exp_args, args, name='RandomIter')


    def active_learn_task(self, task_stream, i):
        """
        active_learn_task: Selects the next few samples to be labelled randomly in multiple iter.
        """

        task = task_stream.tasks[i]
        x_train = task[0][0]
        n_samples_this_task = x_train.shape[0]
        
        n_samples_per_al_cycle = self.get_n_samples_per_al_cycle(n_samples_this_task)
    
        idx_unlabelled = np.arange(n_samples_this_task)
        
        
        for alc in range(self.al_budget):
            print('Iteration:', alc)            

            np.random.shuffle(idx_unlabelled)
            # label data by random sampling n_samples_per_al_cycle number of samples
            labelled_idxs = idx_unlabelled[:n_samples_per_al_cycle]
            # update the unlabelled data
            idx_unlabelled = idx_unlabelled[n_samples_per_al_cycle:]

            new_task = (alc == 0)
            self.agent.learn_task(task, labelled_idxs, new_task, self.args)
