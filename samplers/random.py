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


    def active_learn_task(self, task_stream, i):
        """
        active_learn_task: Selects the next few samples to be labelled randomly.
        """
        
        task = task_stream.tasks[i]
        x_train = task[0][0]
        n_samples_per_al_cycle = self.get_n_samples_per_al_cycle(x_train.shape[0])
        
        idx_unlabelled = np.arange(x_train.shape[0])
        np.random.shuffle(idx_unlabelled)
        
        n_select = self.al_budget * n_samples_per_al_cycle

        self.agent.learn_task(
            task,
            idx_unlabelled[:n_select],
            new_task=True,
            args=self.args)

                    