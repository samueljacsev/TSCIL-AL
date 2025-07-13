# -*- coding: UTF-8 -*-
from types import SimpleNamespace
from agents.base import BaseLearner
from samplers.base import BaseSampler
from utils.data import Dataloader_from_numpy
from sklearn.cluster import KMeans
import numpy as np
import torch


class TypiClustSampler(BaseSampler):
    """
    TypiClustSampler: A sampler implementing the TypiClust strategy for 2-class tasks.
    """

    def __init__(self,
                 agent: BaseLearner,
                 exp_args: SimpleNamespace,
                 args: SimpleNamespace):
        super().__init__(agent, exp_args, args, name='TypiClust')

    def compute_typicality(self, features, k_sym):
        """
        Compute typicality for a batch of features.

        Args:
            features: Feature representations of the data.
            k_sym: Number of nearest neighbors to consider (should match n_samples_per_al_cycle).

        Returns:
            typicalities: Array of typicality scores.
        """
        from sklearn.metrics.pairwise import euclidean_distances

        # Compute pairwise distances
        distances = euclidean_distances(features, features)
        # Sort distances to find k_sym nearest neighbors
        sorted_distances = np.sort(distances, axis=1)[:, 1:k_sym+1]  # Exclude self-distance
        # Compute typicality as the inverse of the average distance to k_sym nearest neighbors
        typicalities = 1 / (np.mean(sorted_distances, axis=1) + 1e-10)

        return typicalities

    def active_learn_task(self, run, task_stream, task_i):
        """
        Selects the next few samples to be labeled based on the TypiClust strategy.

        Args:
            task_stream: Task stream containing tasks.
            task_i: Index of the current task.
        """
        # Set random seeds for reproducibility
        np.random.seed(self.args.seed + run)  # For NumPy operations
        torch.manual_seed(self.args.seed + run)  # For PyTorch operations
        torch.cuda.manual_seed_all(self.args.seed + run)  # For PyTorch CUDA operations (if using GPU)

        task = task_stream.tasks[task_i]
        (x_train, y_train) = task[0]  # y_train is not used for 'unlabeled' data

        n_samples_current_task = x_train.shape[0]
        print('Number of samples in current task:', n_samples_current_task)

        n_samples_per_al_cycle = self.get_n_samples_per_al_cycle(n_samples_current_task)
        n_clusters = n_samples_per_al_cycle

        # Initialize unlabeled indices
        idx_unlabeled = np.arange(n_samples_current_task)

        # Step 1: Representation Learning
        print("Step 1: Representation Learning")
        # Use the agent's model to extract features
        eval_dataloader = Dataloader_from_numpy(
            x_train,
            np.zeros(len(x_train)),  # Dummy labels
            self.batch_size,
            shuffle=False
        )

        all_features = []
        for batch_id, (batch_x, _) in enumerate(eval_dataloader):
            batch_x = batch_x.to(self.agent.device)
            with torch.no_grad():
                features = self.agent.model.feature(batch_x)  # Use the feature method
            all_features.append(features.cpu().numpy())
        all_features = np.vstack(all_features)  # Combine all batches

        for alc in range(self.al_budget):
            print(f'Run: {run}, Task: {task_i}, AL cycle: {alc + 1} / {self.al_budget}')

            if alc == 0:
                # Randomly select the first batch of samples
                np.random.seed(run)  # Set random seed for shuffling
                np.random.shuffle(idx_unlabeled)
                selected_idxs = idx_unlabeled[:n_samples_per_al_cycle]
            else:
                # Step 2: Clustering for Diversity
                print("Step 2: Clustering for Diversity")
                n_clusters = min(n_clusters, len(idx_unlabeled))
                kmeans = KMeans(n_clusters=n_clusters, random_state=self.args.seed + run)  # Set random state for KMeans
                cluster_labels = kmeans.fit_predict(all_features[idx_unlabeled])

                # Step 3: Querying Typical Examples
                print("Step 3: Querying Typical Examples")
                typicalities = self.compute_typicality(all_features[idx_unlabeled], n_samples_per_al_cycle)

                # Select the most typical example from each cluster
                selected_idxs = []
                for cluster in range(n_clusters):
                    cluster_indices = np.where(cluster_labels == cluster)[0]
                    cluster_typicalities = typicalities[cluster_indices]
                    most_typical_idx = cluster_indices[np.argmax(cluster_typicalities)]
                    selected_idxs.append(idx_unlabeled[most_typical_idx])

                # Limit the number of selected samples to the budget per cycle
                selected_idxs = np.array(selected_idxs[:n_samples_per_al_cycle])

            # Update unlabeled indices
            idx_unlabeled = np.setdiff1d(idx_unlabeled, selected_idxs)

            new_task = (alc == 0)  # First cycle is a new task
            # Train the agent on the newly labeled data
            self.agent.learn_task(task, selected_idxs, new_task)
            accuracies = self.agent.evaluate(task_stream, alc, self.al_budget)
            self.save_acc_to_csv(accuracies, run, task_i, alc)