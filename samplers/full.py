# -*- coding: UTF-8 -*-
from types import SimpleNamespace
from agents.base import BaseLearner
from samplers.base import BaseSampler
import numpy as np


class FullSampler(BaseSampler):
    """
    RandomIterSampler: A sampler that selects the full set of samples from the unlabelled pool.
    This is a simple sampler that does not perform any active learning.
    """

    def __init__(self, 
                 agent: BaseLearner, 
                 exp_args: SimpleNamespace, 
                 args: SimpleNamespace ):
        super().__init__(agent, exp_args, args, name='Full')


    def active_learn_task(self, task_stream, i):
        """
        active_learn_task: Selects the full set of samples to be labelled.
        Does not perform any active learning.
        """

        task = task_stream.tasks[i]
        labelled_idxs = np.arange(task[0][0].shape[0])
        
        print('Labelled set size:', len(labelled_idxs))
        self.agent.learn_task(task=task, idxs=labelled_idxs, new_task=True, args=self.args)
