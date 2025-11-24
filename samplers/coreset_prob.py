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
                 probabilistic: bool = True,
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
        if probabilistic:
            self.name += '_probabilistic'

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
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return np.array(idxs, dtype=int)

    def _probabilistic_furthest(self, unlabeled_features, labeled_features, n, random_state):
        """
        Probabilistic k-means++ style sampling for k-center problem.
        
        Instead of greedily picking the furthest point, samples multiple candidates
        proportional to their squared distance and selects the one that minimizes
        total potential. Provides more exploration than deterministic furthest-first.
        
        Args:
            unlabeled_features: Feature vectors of unlabeled samples (m, d).
            labeled_features: Feature vectors of labeled samples (l, d).
            n: Number of samples to select.
            random_state: Random state for reproducibility.
            
        Returns:
            idxs: Indices of selected samples in unlabeled_features.
        """
        m = unlabeled_features.shape[0]
        n_features = unlabeled_features.shape[1]
        
        # Set the number of local trials if not specified
        n_local_trials = self.n_local_trials
        if n_local_trials is None:
            n_local_trials = 2 + int(np.log(n))
        
        # Compute squared norms for efficient distance computation
        unlabeled_squared_norms = (unlabeled_features ** 2).sum(axis=1)
        
        # Initialize minimum squared distances
        if labeled_features.shape[0] == 0:
            # No labeled samples yet - pick first point randomly
            first_idx = random_state.randint(m)
            selected_idxs = [first_idx]
            
            # Calculate distances from first selected point
            closest_dist_sq = euclidean_distances(
                unlabeled_features[[first_idx], :],
                unlabeled_features,
                Y_norm_squared=unlabeled_squared_norms,
                squared=True
            )[0]
            
            start_idx = 1
        else:
            # Use labeled samples as starting points
            labeled_squared_norms = (labeled_features ** 2).sum(axis=1)
            
            # Compute distances from unlabeled to all labeled samples
            dist_to_labeled = euclidean_distances(
                labeled_features,
                unlabeled_features,
                X_norm_squared=labeled_squared_norms,
                Y_norm_squared=unlabeled_squared_norms,
                squared=True
            )
            
            # For each unlabeled sample, track minimum squared distance to labeled set
            closest_dist_sq = np.amin(dist_to_labeled, axis=0)
            
            selected_idxs = []
            start_idx = 0
        
        # Select n samples using probabilistic k-means++ approach
        for i in range(start_idx, n):
            if i % 100 == 0 and i > 0:
                print(f'  Probabilistic selection: {i}/{n}')
            
            # Calculate current potential (sum of squared distances)
            current_pot = closest_dist_sq.sum()
            
            # Sample candidates proportional to squared distance
            rand_vals = random_state.random_sample(n_local_trials) * current_pot
            candidate_idxs = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
            
            # Clip to valid range
            np.clip(candidate_idxs, 0, m - 1, out=candidate_idxs)
            
            # Compute distances from candidates to all unlabeled samples
            candidates_features = unlabeled_features[candidate_idxs]
            distance_to_candidates = euclidean_distances(
                candidates_features,
                unlabeled_features,
                Y_norm_squared=unlabeled_squared_norms,
                squared=True
            )
            
            # Update minimum distances for each candidate
            updated_dist = np.minimum(closest_dist_sq, distance_to_candidates)
            
            # Calculate potential for each candidate (sum of squared distances)
            candidates_pot = updated_dist.sum(axis=1)
            
            # Select candidate that minimizes total potential
            best_candidate_idx = np.argmin(candidates_pot)
            best_idx = candidate_idxs[best_candidate_idx]
            
            selected_idxs.append(best_idx)
            
            # Update closest distances with the newly selected point
            closest_dist_sq = updated_dist[best_candidate_idx]
        
        return np.array(selected_idxs, dtype=int)

    def active_learn_sampler(self, run, task_stream, task_i):
        """Execute CoreSet active learning strategy."""
        accuracies = np.array([])
        task_buffer = np.array([], dtype=int)
        
        # Create random state for probabilistic mode
        if self.probabilistic:
            random_state = np.random.RandomState(self.random_state)
        
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
            
            print(f"CoreSet ({'probabilistic' if self.probabilistic else 'deterministic'}): "
                  f"Selecting from {len(self.idx_unlabeled)} unlabeled samples")
            print(f"CoreSet: Current labeled set size: {len(self.idx_labeled)}")
            
            # Select samples based on mode
            if self.probabilistic:
                # Probabilistic k-means++ sampling
                local_indices = self._probabilistic_furthest(
                    unlabeled_features,
                    labeled_features,
                    self.n_samples_per_al_cycle,
                    random_state
                )
            else:
                # Deterministic greedy furthest-first
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
