# -*- coding: UTF-8 -*-
from types import SimpleNamespace
from agents.base import BaseLearner
from samplers.base import BaseSampler
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import stable_cumsum


class CoreSetProbSampler(BaseSampler):
    """
    CoreSetProbSampler: A probabilistic variant of CoreSet sampling.
    
    Instead of greedily selecting the furthest sample (standard CoreSet), this sampler:
    1. Samples multiple candidates proportional to their squared distance from labeled set
    2. Among candidates, selects the one that best improves coverage (minimizes potential)
    
    This combines CoreSet's diversity objective with kmeans++'s probabilistic robustness,
    making it less sensitive to outliers while maintaining good coverage.
    
    Key differences from deterministic CoreSet:
    - Uses D²-weighted probabilistic sampling instead of always picking furthest point
    - Evaluates multiple candidate samples per selection (n_local_trials)
    - More robust to outliers and noise in the feature space
    
    References:
    - CoreSet: "Active Learning for Convolutional Neural Networks: A Core-Set Approach"
      (Sener & Savarese, ICLR 2018)
    - D²-weighting: k-means++ initialization (Arthur & Vassilvitskii, 2007)
    """

    def __init__(self,
                 agent: BaseLearner,
                 exp_args: SimpleNamespace,
                 args: SimpleNamespace,
                 n_local_trials: int = None):
        """
        Args:
            agent: The learning agent
            exp_args: Experiment arguments
            args: Sampler arguments
            n_local_trials: Number of candidate trials per selection.
                          Default: 2 + log(n_samples_per_cycle)
        """
        super().__init__(agent, exp_args, args, name='CoreSetProb')
        self.n_local_trials = n_local_trials
        print(f"CoreSetProb initialized with D²-weighted probabilistic sampling")

    def probabilistic_furthest_first(self, unlabeled_features, labeled_features, n, random_state):
        """
        Probabilistic k-means++ style sampling for k-center problem.
        
        Instead of greedily picking the furthest point, this method:
        1. Samples n_local_trials candidates proportional to squared distance (D² weighting)
        2. Evaluates each candidate's potential (total remaining coverage)
        3. Selects the candidate with minimum potential (best coverage improvement)
        
        This provides robustness to outliers while maintaining diversity.
        
        Args:
            unlabeled_features: Feature vectors of unlabeled samples (m, d).
            labeled_features: Feature vectors of labeled samples (l, d).
            n: Number of samples to select.
            random_state: Random state for reproducibility.
            
        Returns:
            idxs: Indices of selected samples in unlabeled_features.
        """
        m = unlabeled_features.shape[0]
        
        # Set the number of local trials if not specified
        n_local_trials = self.n_local_trials
        if n_local_trials is None:
            n_local_trials = max(2 + int(np.log(n)), 1)
        
        print(f"CoreSetProb: Using {n_local_trials} trials per selection")
        
        # Compute squared norms for efficient distance computation
        unlabeled_squared_norms = (unlabeled_features ** 2).sum(axis=1)
        
        # Initialize minimum squared distances
        if labeled_features.shape[0] == 0:
            # No labeled samples yet - pick first point randomly
            first_idx = random_state.randint(m)
            selected_idxs = [first_idx]
            
            # Calculate squared distances from first selected point
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
            
            # Compute squared distances from unlabeled to all labeled samples
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
        
        # Select n samples using probabilistic D²-weighted sampling
        for i in range(start_idx, n):
            if i % 100 == 0 and i > 0:
                print(f'  CoreSetProb: Selected {i}/{n} samples')
            
            # Calculate current potential (sum of squared distances)
            current_pot = closest_dist_sq.sum()
            
            if current_pot == 0:
                # All samples at zero distance - shouldn't happen but handle gracefully
                print(f"Warning: Zero potential at iteration {i}. Falling back to random selection.")
                # From the unlabeled pool, find indices that have not yet been selected in this batch
                remaining_m_indices = np.setdiff1d(np.arange(m), selected_idxs)
                if remaining_m_indices.size > 0:
                    idx = random_state.choice(remaining_m_indices)
                    selected_idxs.append(idx)
                continue
            
            # Sample n_local_trials candidates proportional to squared distance (D² weighting)
            rand_vals = random_state.random_sample(n_local_trials) * current_pot
            candidate_idxs = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
            
            # Clip to valid range
            np.clip(candidate_idxs, 0, m - 1, out=candidate_idxs)
            
            # Compute squared distances from each candidate to all unlabeled samples
            candidates_features = unlabeled_features[candidate_idxs]
            distance_to_candidates = euclidean_distances(
                candidates_features,
                unlabeled_features,
                Y_norm_squared=unlabeled_squared_norms,
                squared=True
            )
            
            # For each candidate, compute what the new minimum distances would be
            updated_dist = np.minimum(closest_dist_sq, distance_to_candidates)
            
            # Calculate potential for each candidate (total remaining squared distance)
            candidates_pot = updated_dist.sum(axis=1)
            
            # Select candidate that minimizes total potential (best coverage)
            best_candidate_idx = np.argmin(candidates_pot)
            best_idx = candidate_idxs[best_candidate_idx]
            
            selected_idxs.append(best_idx)
            
            # Update closest distances with the newly selected point
            closest_dist_sq = updated_dist[best_candidate_idx]
            
            if i < 5:
                print(f"  Selected sample {i}: index {best_idx}, new potential: {candidates_pot[best_candidate_idx]:.2f}")
        
        return np.array(selected_idxs, dtype=int)

    def active_learn_sampler(self, run, task_stream, task_i):
        """Execute CoreSetProb active learning strategy."""
        accuracies = np.array([])
        task_buffer = np.array([], dtype=int)
        
        
        for alc in range(self.al_budget):
            print(f'AL cycle: {alc + 1} / {self.al_budget}')

                    # Create random state for reproducibility
            random_state = np.random.RandomState(self.random_state + alc)

            # Extract features from all training samples
            x_train, _ = self.current_task[0]
            all_features, _ = self.extract_features_and_outputs(x_train)
            
            # Get features for labeled and unlabeled sets
            unlabeled_features = all_features[self.idx_unlabeled]
            
            if self.idx_labeled.size > 0:
                labeled_features = all_features[self.idx_labeled]
            else:
                labeled_features = np.empty((0, all_features.shape[1]))
            
            print(f"CoreSetProb: Selecting from {len(self.idx_unlabeled)} unlabeled samples")
            print(f"CoreSetProb: Current labeled set size: {len(self.idx_labeled)}")
            
            # Apply probabilistic furthest-first sampling
            local_indices = self.probabilistic_furthest_first(
                unlabeled_features,
                labeled_features,
                self.n_samples_per_al_cycle,
                random_state
            )
            
            selected_idxs = self.idx_unlabeled[local_indices]
            
            print(f"CoreSetProb: Selected {len(selected_idxs)} samples")

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
