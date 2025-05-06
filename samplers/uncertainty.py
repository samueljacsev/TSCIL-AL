# -*- coding: UTF-8 -*-
from types import SimpleNamespace
from agents.base import BaseLearner
from samplers.base import BaseSampler
from utils.data import Dataloader_from_numpy
import numpy as np
import torch
from sklearn.cluster import KMeans


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

    
   
    


    
    # def active_learn_task(self, run, task_stream, task_i, metric='least_confidence'):
    #     """
    #     Selects the next few samples to be labelled based on uncertainty sampling.

    #     Args:
    #         task_stream: Task stream containing tasks.
    #         i: Index of the current task.
    #         metric: Uncertainty metric ('entropy', 'margin', 'least_confidence').
    #     """
    #     if self.args.uncertainty_type is not None:
    #         metric = self.args.uncertainty_type
        
    #     task = task_stream.tasks[task_i]
    #     (x_train, y_train) = task[0]  # y_train is not used for unlabeled data

    #     n_samples_current_task = x_train.shape[0]
    #     print('Number of samples in current task:', n_samples_current_task)
        
    #     n_samples_per_al_cycle = self.get_n_samples_per_al_cycle(n_samples_current_task)

    #     # Initialize unlabeled indices
    #     idx_unlabeled = np.arange(n_samples_current_task)

    #     for alc in range(self.al_budget):
    #         print(f'Run: {run}, Task: {task_i}, AL cycle: {alc + 1} / {self.al_budget}')

    #         if alc == 0:
    #             # Randomly select the first batch of samples
    #             np.random.shuffle(idx_unlabeled)
    #             selected_idxs = idx_unlabeled[:n_samples_per_al_cycle]
    #         else:
    #             # Evaluate the model on the unlabeled data
    #             eval_dataloader = Dataloader_from_numpy(
    #                 x_train[idx_unlabeled], 
    #                 np.zeros(len(idx_unlabeled)),  # Dummy labels
    #                 self.batch_size,
    #                 shuffle=False
    #             )

    #             # Collect outputs for all unlabeled samples
    #             all_outputs = []
    #             for batch_id, (batch_x, _) in enumerate(eval_dataloader):
    #                 batch_x = batch_x.to(self.agent.device)
    #                 with torch.no_grad():
    #                     outputs = self.agent.model(batch_x)  # Forward pass
    #                 all_outputs.append(outputs)
    #             all_outputs = torch.cat(all_outputs, dim=0)  # Combine all batches

    #             # Compute uncertainty scores
    #             uncertainties = self.compute_uncertainty(all_outputs, metric=metric)

    #             # Select the top uncertain samples
    #             selected_idxs = idx_unlabeled[np.argsort(-uncertainties)[:n_samples_per_al_cycle]]

    #         # Update unlabeled indices
    #         idx_unlabeled = np.setdiff1d(idx_unlabeled, selected_idxs)

    #         new_task = (alc == 0)  # First cycle is a new task
    #         # Train the agent on the newly labelled data
    #         self.agent.learn_task(task, selected_idxs, new_task)
    #         accuracies = self.agent.evaluate(task_stream, alc, self.al_budget)
    #         self.save_acc_to_csv(accuracies, run, task_i, alc, f'_{metric}')
    


    def active_learn_task(self, run, task_stream, task_i, metric='least_confidence', ood_indices=None, ood_scores=None):
            if self.args.uncertainty_type is not None:
                metric = self.args.uncertainty_type

            task = task_stream.tasks[task_i]
            (x_train, y_train) = task[0]

            n_samples_current_task = x_train.shape[0]
            n_samples_per_al_cycle = self.get_n_samples_per_al_cycle(n_samples_current_task)

            idx_unlabeled = np.arange(n_samples_current_task)
            task_features = []

            for alc in range(self.al_budget):
                print(f'Run: {run}, Task: {task_i}, AL cycle: {alc + 1} / {self.al_budget}')
                if alc == 0:
                    if task_i == 0:
                        np.random.shuffle(idx_unlabeled)
                        selected_idxs = idx_unlabeled[:n_samples_per_al_cycle]
                    else:
                        selected_idxs = self.ood_filter_top_ood(
                            x_train,
                            y_train,
                            idx_unlabeled,
                            task_stream,
                            n_samples_per_al_cycle,
                            run,
                            task_i,
                            ood_scores
                        )
                else:
                    eval_dataloader = Dataloader_from_numpy(
                        x_train[idx_unlabeled],
                        np.zeros(len(idx_unlabeled)),
                        self.batch_size,
                        shuffle=False
                    )

                    all_outputs = []
                    for batch_id, (batch_x, _) in enumerate(eval_dataloader):
                        batch_x = batch_x.to(self.agent.device)
                        with torch.no_grad():
                            outputs = self.agent.model(batch_x)
                        all_outputs.append(outputs)
                    all_outputs = torch.cat(all_outputs, dim=0)

                    uncertainties = self.compute_uncertainty(all_outputs, metric=metric)
                    selected_idxs = idx_unlabeled[np.argsort(-uncertainties)[:n_samples_per_al_cycle]]

                idx_unlabeled = np.setdiff1d(idx_unlabeled, selected_idxs)

                new_task = (alc == 0)
                self.agent.learn_task(task, selected_idxs, new_task)
                accuracies = self.agent.evaluate(task_stream, alc, self.al_budget)
                self.save_acc_to_csv(accuracies, run, task_i, alc, f'_{metric}', ood_method='energy')

                labeled_indices = np.setdiff1d(np.arange(n_samples_current_task), idx_unlabeled)
                if len(labeled_indices) > 0:
                    eval_dataloader = Dataloader_from_numpy(
                        x_train[labeled_indices],
                        np.zeros(len(labeled_indices)),
                        self.batch_size,
                        shuffle=False
                    )

                    all_features = []
                    for batch_id, (batch_x, _) in enumerate(eval_dataloader):
                        batch_x = batch_x.to(self.agent.device)
                        with torch.no_grad():
                            features = self.agent.model.feature(batch_x)
                        all_features.append(features.cpu().numpy())
                    all_features = np.vstack(all_features)
                    task_features.append(all_features)

            if task_features:
                task_features_combined = np.vstack(task_features)
                self.id_features_list.append(task_features_combined)
                print(f"Task {task_i} - Stored {len(task_features_combined)} features in id_features_list.")

            return idx_unlabeled
            
