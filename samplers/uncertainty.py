# -*- coding: UTF-8 -*-
from types import SimpleNamespace
from agents.base import BaseLearner
from samplers.base import BaseSampler
from utils.data import Dataloader_from_numpy
import numpy as np


class UncertaintySampler(BaseSampler):
    """
    UncertaintySampler: A sampler with uncertainty sampling strategy.
    """

    def __init__(self, agent: BaseLearner, exp_args: SimpleNamespace, args: SimpleNamespace, ):
        super().__init__(agent, exp_args, args, name='Uncertainty')


        
    def active_learn_task(self, task_stream, i):
        """
        active_learn_task: Selects the next few samples to be labelled based on uncertainty sampling.
        """

        n_train_samples_per_task = self.get_n_train_samples_per_task()
        n_samples_per_al_cycle = int(n_train_samples_per_task / self.al_total)
        
        task = task_stream.tasks[i]
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = task
        idx_unlabelled = np.arange(x_train.shape[0])
        
        
        for alc in range(self.al_budget):
            print('Iteration:', alc)            

            if alc == 0:
                np.random.shuffle(idx_unlabelled)
                # label data by random sampling n_samples_per_al_cycle number of samples
                labelled_idxs = idx_unlabelled[:n_samples_per_al_cycle]
                # update the unlabelled data
                idx_unlabelled = idx_unlabelled[n_samples_per_al_cycle:]
            else:
                # predict the uncertainty of the unlabelled data          
                eval_dataloader_i = Dataloader_from_numpy(
                    x_train[idx_unlabelled], y_train[idx_unlabelled], self.batch_size, shuffle=False)
                print("Evaluate the model on the current task")
                self.agent.evaluate_on_dataloader(eval_dataloader_i, task_stream, i)

            if alc == 0:
                new_task = True
            else:
                new_task = False
                
            self.agent.learn_task(task, labelled_idxs, new_task, self.args)
