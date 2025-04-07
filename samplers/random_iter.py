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

        n_train_samples_per_task = self.get_n_train_samples_per_task()
        n_samples_per_al_cycle = int(n_train_samples_per_task / self.al_total)
        
        task = task_stream.tasks[i]
        (x_train, y_train) = task[0]
        idx_unlabelled = np.arange(x_train.shape[0])
        
        
        for alc in range(self.al_budget):
            print(f'AL cycle: {alc} / {self.al_budget}')            

            np.random.shuffle(idx_unlabelled)
            # label data by random sampling n_samples_per_al_cycle number of samples
            labelled_idxs = idx_unlabelled[:n_samples_per_al_cycle]
            # update the unlabelled data
            idx_unlabelled = idx_unlabelled[n_samples_per_al_cycle:]
            
            new_task = (alc == 0) # First cycle is a new task
            
            print('Labelled set size:', len(labelled_idxs))
            self.agent.learn_task(task, labelled_idxs, new_task, self.args)
