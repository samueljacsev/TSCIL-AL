# -*- coding: UTF-8 -*-
from types import SimpleNamespace
from agents.base import BaseLearner
from samplers.base import BaseSampler
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import stable_cumsum


class KMeansPPPSampler(BaseSampler):
    """
    KMeansPPPSampler: A sampler implementing diversity-based k-means++ sampling for active learning.
    Adapted from dm.py to work with active learning cycles.
    """

    def __init__(self,
                 agent: BaseLearner,
                 exp_args: SimpleNamespace,
                 args: SimpleNamespace):
        super().__init__(agent, exp_args, args, name='kmeans_ppp')

    def _calc_starting_state(self, X, starting_points, x_squared_norms, score_weight):
        """
        Calculate initial distance state based on starting points (labeled samples).
        
        Args:
            X: Feature matrix for all samples
            starting_points: Indices of labeled samples
            x_squared_norms: Squared norms of feature vectors
            score_weight: Weight array for scoring
            
        Returns:
            closest_dist_sq: Minimum squared distances to any starting point
        """
        closest_dist_sq = euclidean_distances(
            X[starting_points[0], np.newaxis], X, 
            Y_norm_squared=x_squared_norms, squared=True
        ) * score_weight
        
        for e, point in enumerate(starting_points[1:]):
            if e % 100 == 0:
                print(f'Calculating starting state at {e} iteration')
            distance_to_candidates = euclidean_distances(
                X[point, np.newaxis], X, 
                Y_norm_squared=x_squared_norms, squared=True
            ) * score_weight
            
            # Update closest distances
            np.minimum(closest_dist_sq, distance_to_candidates,
                      out=distance_to_candidates)
            closest_dist_sq = distance_to_candidates
        
        return closest_dist_sq

    def _kmeans_plusplus_active(self, X, unlabeled_indices, n_clusters, 
                                x_squared_norms, random_state, 
                                n_local_trials=None, starting_points=None, 
                                score_weight=None):
        """
        K-means++ sampling adapted for active learning.
        Selects diverse samples from unlabeled pool using labeled samples as anchors.
        
        Args:
            X: Feature matrix for all samples
            unlabeled_indices: Indices of unlabeled samples
            n_clusters: Number of new samples to select
            x_squared_norms: Squared norms of feature vectors
            random_state: Random state for reproducibility
            n_local_trials: Number of trials per selection (default: 2 + log(n_clusters))
            starting_points: Indices of labeled samples (used as anchors)
            score_weight: Weight array for scoring (default: ones)
            
        Returns:
            centers: Selected feature vectors
            indices: Indices of selected samples in original dataset
        """
        if score_weight is None:
            score_weight = np.ones(X.shape[0])
        else:
            print(f'Mean of score weights: {np.mean(score_weight)}')
        
        n_samples, n_features = X.shape
        
        # Set the number of local seeding trials if none is given
        if n_local_trials is None:
            n_local_trials = 2 + int(np.log(n_clusters))
        
        # Initialize based on whether we have starting points (labeled samples)
        if starting_points is None or len(starting_points) == 0:
            # First AL cycle: no labeled samples yet
            centers = np.empty((n_clusters, n_features), dtype=X.dtype)
            indices = np.full(n_clusters, -1, dtype=int)
            
            # Pick first center randomly from unlabeled pool
            random_idx = random_state.randint(len(unlabeled_indices))
            center_id = unlabeled_indices[random_idx]
            centers[0] = X[center_id]
            indices[0] = center_id
            
            # Calculate initial distances
            closest_dist_sq = euclidean_distances(
                centers[0, np.newaxis], X, 
                Y_norm_squared=x_squared_norms, squared=True
            ) * score_weight
            
            selection_range = range(1, n_clusters)
            
        else:
            # Subsequent AL cycles: use labeled samples as starting points
            print(f'Using {len(starting_points)} labeled samples as starting points')
            closest_dist_sq = self._calc_starting_state(
                X, starting_points, x_squared_norms, score_weight
            )
            
            centers = np.empty((n_clusters, n_features), dtype=X.dtype)
            indices = np.full(n_clusters, -1, dtype=int)
            
            selection_range = range(n_clusters)
        
        # Extract distances for unlabeled samples only
        unlabeled_dist_sq = closest_dist_sq[0, unlabeled_indices]
        current_pot = unlabeled_dist_sq.sum()
        
        # Select n_clusters samples from unlabeled pool
        for c in selection_range:
            if c % 100 == 0:
                print(f'Selecting sample {c}/{n_clusters}')
            
            # Sample candidates from unlabeled pool proportional to squared distance
            rand_vals = random_state.random_sample(n_local_trials) * current_pot
            candidate_local_ids = np.searchsorted(
                stable_cumsum(unlabeled_dist_sq), rand_vals
            )
            
            # Clip to valid range
            np.clip(candidate_local_ids, None, len(unlabeled_indices) - 1,
                   out=candidate_local_ids)
            
            # Map back to original indices
            candidate_ids = unlabeled_indices[candidate_local_ids]
            
            # Compute distances from candidates to all samples
            distance_to_candidates = euclidean_distances(
                X[candidate_ids], X, 
                Y_norm_squared=x_squared_norms, squared=True
            ) * score_weight
            
            # Update closest distances
            np.minimum(closest_dist_sq, distance_to_candidates,
                      out=distance_to_candidates)
            
            # Calculate potential for each candidate
            candidates_pot = distance_to_candidates[:, unlabeled_indices].sum(axis=1)
            
            # Select best candidate (minimizes total potential)
            best_candidate_local = np.argmin(candidates_pot)
            best_candidate = candidate_ids[best_candidate_local]
            
            # Update state
            current_pot = candidates_pot[best_candidate_local]
            closest_dist_sq = distance_to_candidates[best_candidate_local:best_candidate_local+1]
            unlabeled_dist_sq = closest_dist_sq[0, unlabeled_indices]
            
            # Store selected center
            centers[c] = X[best_candidate]
            indices[c] = best_candidate
            
            if c < 10:
                print(f'Selected sample {c}: index {best_candidate}')
        
        return centers, indices

    def active_learn_sampler(self, run, task_stream, task_i):
        """Execute kmeans_ppp (diversity-based k-means++) active learning strategy."""
        accuracies = np.array([])
        task_buffer = np.array([], dtype=int)
        
        for alc in range(self.al_budget):
            print(f'AL cycle: {alc + 1} / {self.al_budget}')
            
            # Extract features from training data
            x_train, _ = self.current_task[0]
            all_features, _ = self.extract_features_and_outputs(x_train)
            
            # Prepare parameters for k-means++
            x_squared_norms = (all_features ** 2).sum(axis=1)
            n_clusters = self.n_samples_per_al_cycle
            
            # Use labeled samples as starting points (from 2nd cycle onwards)
            starting_points = self.idx_labeled if alc > 0 else None
            
            # Create random state for this cycle
            random_state = np.random.RandomState(self.random_state + alc)
            
            print(f"Selecting {n_clusters} samples using k-means++ diversity sampling")
            if starting_points is not None:
                print(f"Using {len(starting_points)} labeled samples as anchors")
            
            # Run k-means++ selection on unlabeled samples
            _, selected_idxs = self._kmeans_plusplus_active(
                X=all_features,
                unlabeled_indices=self.idx_unlabeled,
                n_clusters=n_clusters,
                x_squared_norms=x_squared_norms,
                random_state=random_state,
                starting_points=starting_points,
                score_weight=None  # Default to ones
            )
            
            print(f'Number of selected indices: {len(selected_idxs)}')
            
            # Update labeled and unlabeled sets
            self.idx_labeled = np.concatenate([self.idx_labeled, selected_idxs])
            self.idx_unlabeled = np.setdiff1d(self.idx_unlabeled, selected_idxs,
                                             assume_unique=True)
            
            # Add selected samples to the task buffer
            task_buffer = np.concatenate([task_buffer, selected_idxs])
            
            # Train and evaluate
            self.agent.learn_task(self.current_task, task_buffer, alc == 0)
            accuracies = self.agent.evaluate(task_stream, alc, self.al_budget)
            self.save_acc_to_csv(accuracies, run, task_i, alc)
        
        return accuracies
