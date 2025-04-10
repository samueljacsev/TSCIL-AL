# -*- coding: UTF-8 -*-
from types import SimpleNamespace
from agents.base import BaseLearner
from samplers.base import BaseSampler
from utils.data import Dataloader_from_numpy
import numpy as np
import torch


class UncertaintySampler(BaseSampler):
    """
    UncertaintySampler: A sampler with uncertainty sampling strategy.
    """

    def __init__(self, 
                 agent: BaseLearner, 
                 exp_args: SimpleNamespace, 
                 args: SimpleNamespace):
        super().__init__(agent, exp_args, args, name='Uncertainty')


    def compute_uncertainty(self, outputs, metric='least_confidence'):
        """
        Compute uncertainty for a batch of outputs.

        Args:
            outputs: Model outputs (logits or probabilities).
            metric: Uncertainty metric ('entropy', 'margin', 'least_confidence').

        Returns:
            uncertainties: Array of uncertainty scores.
        """
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        if metric == 'entropy':
            # Compute entropy
            uncertainties = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        elif metric == 'margin':
            # Compute margin (difference between top-2 probabilities)
            sorted_probs = -np.sort(-probabilities, axis=1)  # Sort in descending order
            uncertainties = 1 - (sorted_probs[:, 0] - sorted_probs[:, 1])
        elif metric == 'least_confidence':
            # Compute least confidence (1 - max probability)
            uncertainties = 1 - np.max(probabilities, axis=1)
        else:
            raise ValueError(f"Unknown uncertainty metric: {metric}")

        return uncertainties

    
    def active_learn_task(self, run, task_stream, task_i, metric='least_confidence'):
        """
        Selects the next few samples to be labelled based on uncertainty sampling.

        Args:
            task_stream: Task stream containing tasks.
            i: Index of the current task.
            metric: Uncertainty metric ('entropy', 'margin', 'least_confidence').
        """
        task = task_stream.tasks[task_i]
        (x_train, y_train) = task[0]  # y_train is not used for unlabelled data

        n_samples_current_task = x_train.shape[0]
        print('Number of samples in current task:', n_samples_current_task)
        
        n_samples_per_al_cycle = self.get_n_samples_per_al_cycle(n_samples_current_task)

        # Initialize unlabelled indices
        idx_unlabelled = np.arange(n_samples_current_task)

        for alc in range(self.al_budget):
            print(f'AL cycle: {alc + 1} / {self.al_budget}')

            if alc == 0:
                # Randomly select the first batch of samples
                np.random.shuffle(idx_unlabelled)
                selected_idxs = idx_unlabelled[:n_samples_per_al_cycle]
            else:
                print('Shape of unlabelled data:', x_train[idx_unlabelled].shape)

                # Evaluate the model on the unlabelled data
                eval_dataloader = Dataloader_from_numpy(
                    x_train[idx_unlabelled], 
                    np.zeros(len(idx_unlabelled)),  # Dummy labels
                    self.batch_size,
                    shuffle=False
                )

                # Collect outputs for all unlabelled samples
                all_outputs = []
                for batch_id, (batch_x, _) in enumerate(eval_dataloader):
                    batch_x = batch_x.to(self.agent.device)
                    with torch.no_grad():
                        outputs = self.agent.model(batch_x)  # Forward pass
                    all_outputs.append(outputs)
                all_outputs = torch.cat(all_outputs, dim=0)  # Combine all batches

                # Compute uncertainty scores
                uncertainties = self.compute_uncertainty(all_outputs, metric=metric)

                # Select the top uncertain samples
                selected_idxs = idx_unlabelled[np.argsort(-uncertainties)[:n_samples_per_al_cycle]]

            # Update unlabelled indices
            idx_unlabelled = np.setdiff1d(idx_unlabelled, selected_idxs)

            new_task = (alc == 0)  # First cycle is a new task
            # Train the agent on the newly labelled data
            self.agent.learn_task(task, selected_idxs, new_task)
            accuracies = self.agent.evaluate(run, task_stream, task_i, alc, self.al_budget)
            self.save_acc_to_csv(accuracies, run, task_i, alc, f'_{metric}')
            