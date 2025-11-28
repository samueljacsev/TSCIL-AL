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
        super().__init__(agent, exp_args, args, name='Random')


    def active_learn_sampler(self, run, task_stream, task_i):
        """Execute random sampling active learning strategy."""
        accuracies = np.array([])
        task_buffer = np.array([], dtype=int)
        
        for alc in range(self.al_budget):
            print(f'AL cycle: {alc + 1} / {self.al_budget}')

            # Randomly select samples from unlabeled pool
            np.random.shuffle(self.idx_unlabeled)
            selected_idxs = self.idx_unlabeled[:self.n_samples_per_al_cycle].copy()
            print(f'Number of selected indices: {len(selected_idxs)}')

            # Update labeled and unlabeled sets
            self.idx_labeled = np.concatenate([self.idx_labeled, selected_idxs])
            self.idx_unlabeled = self.idx_unlabeled[self.n_samples_per_al_cycle:]

            # Add selected samples to the task buffer
            task_buffer = np.concatenate([task_buffer, selected_idxs])

            # Train and evaluate
            self.agent.learn_task(self.current_task, task_buffer, alc == 0)
            accuracies = self.agent.evaluate(task_stream, alc, self.al_budget)
            self.save_acc_to_csv(accuracies, run, task_i, alc)

        return accuracies
            
