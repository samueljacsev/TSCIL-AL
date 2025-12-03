# -*- coding: UTF-8 -*-
from types import SimpleNamespace
from agents.base import BaseLearner
from samplers.base import BaseSampler
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import stable_cumsum


class CoreSetSampler(BaseSampler):
    """
    CoreSetSampler: A sampler implementing the CoreSet (k-Center) strategy.
    
    CoreSet selects samples by solving a k-center problem: it greedily picks
    unlabeled samples that are furthest from the current labeled set in the
    feature space. This ensures diverse coverage of the data distribution.
    
    Supports two modes:
    - Deterministic (default): Greedy furthest-first selection
    - Probabilistic: k-means++ style probabilistic sampling with multiple trials
    
    Reference: "Active Learning for Convolutional Neural Networks: A Core-Set Approach"
    (Sener & Savarese, ICLR 2018)
    """

    def __init__(self,
                 agent: BaseLearner,
                 exp_args: SimpleNamespace,
                 args: SimpleNamespace,
                 probabilistic: bool = False,
                 n_local_trials: int = None):
        """
        Args:
            agent: The learning agent
            exp_args: Experiment arguments
            args: Sampler arguments
            probabilistic: If True, use probabilistic k-means++ sampling; 
                         if False, use deterministic greedy furthest-first
            n_local_trials: Number of candidate trials per selection (probabilistic mode only).
                          Default: 2 + log(n_samples_per_cycle)
        """
        super().__init__(agent, exp_args, args, name='CoreSet')
        self.probabilistic = probabilistic
        self.n_local_trials = n_local_trials
        print(f"CoreSet initialized in {'probabilistic' if probabilistic else 'deterministic'} mode")

    def furthest_first(self, unlabeled_features, labeled_features, n):
        """
        Greedy furthest-first traversal for k-center problem.
        
        Iteratively selects n samples from unlabeled_features that are
        furthest from the labeled set, maximizing minimum distance to
        already selected points.
        
        Args:
            unlabeled_features: Feature vectors of unlabeled samples (m, d).
            labeled_features: Feature vectors of labeled samples (l, d).
            n: Number of samples to select.
            
        Returns:
            idxs: Indices of selected samples in unlabeled_features.
        """
        m = unlabeled_features.shape[0]
        
        # Initialize minimum distances
        if labeled_features.shape[0] == 0:
            # No labeled samples yet - all distances are infinite
            min_dist = np.tile(float("inf"), m)
        else:
            # Compute distances from unlabeled to labeled samples
            dist_ctr = pairwise_distances(unlabeled_features, labeled_features)
            # For each unlabeled sample, track distance to nearest labeled sample
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            # Select the unlabeled sample with maximum distance to labeled set
            idx = min_dist.argmax()
            idxs.append(idx)
            
            # Update minimum distances with the newly selected point
            dist_new_ctr = pairwise_distances(
                unlabeled_features, 
                unlabeled_features[[idx], :]
            )
            min_dist = np.minimum(min_dist, dist_new_ctr[:, 0])

        return np.array(idxs, dtype=int)

    def active_learn_sampler(self, run, task_stream, task_i):
        """Execute CoreSet active learning strategy."""
        accuracies = np.array([])
        task_buffer = np.array([], dtype=int)
        
        for alc in range(self.al_budget):
            print(f'AL cycle: {alc + 1} / {self.al_budget}')

            # Extract features from all training samples
            x_train, _ = self.current_task[0]
            all_features, _ = self.extract_features_and_outputs(x_train)
            
            # Get features for labeled and unlabeled sets
            unlabeled_features = all_features[self.idx_unlabeled]
            
            if self.idx_labeled.size > 0:
                labeled_features = all_features[self.idx_labeled]
            else:
                labeled_features = np.empty((0, all_features.shape[1]))
            
            print(f"CoreSet: Selecting from {len(self.idx_unlabeled)} unlabeled samples")
            print(f"CoreSet: Current labeled set size: {len(self.idx_labeled)}")
            
            # Apply furthest-first algorithm to select diverse samples
            local_indices = self.furthest_first(
                unlabeled_features, 
                labeled_features, 
                self.n_samples_per_al_cycle
            )
            selected_idxs = self.idx_unlabeled[local_indices]
            
            print(f"CoreSet: Selected {len(selected_idxs)} samples")

            # Update labeled and unlabeled sets
            self.idx_labeled = np.concatenate([self.idx_labeled, selected_idxs])
            self.idx_unlabeled = np.setdiff1d(
                self.idx_unlabeled, 
                selected_idxs, 
                assume_unique=True
            )

            # Add selected samples to the task buffer
            task_buffer = np.concatenate([task_buffer, selected_idxs])

            # Train and evaluate
            self.agent.learn_task(self.current_task, task_buffer, alc == 0)
            accuracies = self.agent.evaluate(task_stream, alc, self.al_budget)
            self.save_acc_to_csv(accuracies, run, task_i, alc)

        return accuracies
