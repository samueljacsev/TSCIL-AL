# -*- coding: UTF-8 -*-
from types import SimpleNamespace
from agents.base import BaseLearner
from samplers.base import BaseSampler
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
        self.metric = args.uncertainty_type if args.uncertainty_type else 'margin'


    def compute_uncertainty(self, outputs):
        """
        Compute uncertainty for a batch of outputs.

        Args:
            outputs: Model outputs (logits or probabilities).

        Returns:
            uncertainties: Array of uncertainty scores.
        """
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        if self.metric == 'entropy':
            # Compute entropy
            uncertainties = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        elif self.metric == 'margin':
            # Compute margin (difference between top-2 probabilities)
            sorted_probs = -np.sort(-probabilities, axis=1)  # Sort in descending order
            uncertainties = 1 - (sorted_probs[:, 0] - sorted_probs[:, 1])
        elif self.metric == 'least_confidence':
            # Compute least confidence (1 - max probability)
            uncertainties = 1 - np.max(probabilities, axis=1)
        else:
            raise ValueError(f"Unknown uncertainty metric: {self.metric}")

        return uncertainties


    def active_learn_sampler(self, run, task_stream, task_i):
        """Execute uncertainty-based active learning strategy."""
        accuracies = np.array([])
        
        for alc in range(self.al_budget):
            print(f'AL cycle: {alc + 1} / {self.al_budget}')

            if alc == 0:
                # Randomly select the first batch of samples
                np.random.shuffle(self.idx_unlabeled)
                selected_idxs = self.idx_unlabeled[:self.n_samples_per_al_cycle].copy()
            else:
                # Extract outputs for unlabeled samples only
                x_train, _ = self.current_task[0]
                x_unlabeled = x_train[self.idx_unlabeled]
                _, unlabeled_outputs = self.extract_features_and_outputs(x_unlabeled)

                # Compute uncertainty scores
                print(f"Computing uncertainty using {self.metric} metric")
                uncertainties = self.compute_uncertainty(unlabeled_outputs)

                # Select the most uncertain samples
                local_indices = np.argsort(-uncertainties)[:self.n_samples_per_al_cycle]
                selected_idxs = self.idx_unlabeled[local_indices]

            # Update labeled and unlabeled sets
            self.idx_labeled = np.concatenate([self.idx_labeled, selected_idxs])
            self.idx_unlabeled = np.setdiff1d(self.idx_unlabeled, selected_idxs, 
                                             assume_unique=True)

            # Train and evaluate
            self.agent.learn_task(self.current_task, selected_idxs, alc == 0)
            accuracies = self.agent.evaluate(task_stream, alc, self.al_budget)
            self.save_acc_to_csv(accuracies, run, task_i, alc, f'_{self.metric}')

        return accuracies