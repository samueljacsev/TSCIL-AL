# -*- coding: UTF-8 -*-
from types import SimpleNamespace
from agents.base import BaseLearner
from samplers.base import BaseSampler
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import time
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import stable_cumsum


class TypiKmeansSampler(BaseSampler):
    """
    TypiKmeansSampler: A hybrid sampler that uses TypiClust for the initial
    exploration phase and switches to CoreSetProb for exploitation.
    
    - First 2 AL cycles: Uses TypiClust to quickly find representative samples
      from dense regions, establishing a good initial labeled set.
    - Subsequent AL cycles: Uses CoreSetProb to select diverse samples that are
      far from the existing labeled set, improving model coverage.
    """
    
    MIN_CLUSTER_SIZE = 3
    MAX_NUM_CLUSTERS = np.inf
    K_NN = 20

    def __init__(self,
                 agent: BaseLearner,
                 exp_args: SimpleNamespace,
                 args: SimpleNamespace,
                 n_local_trials: int = None):
        super().__init__(agent, exp_args, args, name='TypiKmeans')
        self.n_local_trials = n_local_trials
        print(f"TypiKmeans initialized: TypiClust (2 cycles) -> CoreSetProb")

    # --- TypiClust Methods ---
    
    def get_clusters(self, features, n_clusters):
        """Perform K-means clustering. Uses MiniBatchKMeans for scalability."""
        n_clusters = min(n_clusters, len(features), self.MAX_NUM_CLUSTERS)
        
        if n_clusters > 50:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, 
                                    batch_size=min(5000, len(features)),
                                    random_state=self.random_state)
        else:
            kmeans = KMeans(n_clusters=n_clusters, 
                          random_state=self.random_state)
        
        return kmeans.fit_predict(features)

    def _get_nn_sklearn(self, features, num_neighbors):
        """Calculate nearest neighbors using sklearn (CPU fallback)."""
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=num_neighbors + 1, algorithm='auto', metric='euclidean')
        nn.fit(features)
        distances, indices = nn.kneighbors(features)
        
        return distances[:, 1:], indices[:, 1:]

    def compute_typicality(self, features, k_nn=None):
        """Compute typicality scores based on k-NN distances."""
        if k_nn is None:
            k_nn = min(self.K_NN, len(features) // 2)
        
        k_nn = max(1, min(k_nn, len(features) - 1))
        
        distances, _ = self._get_nn_sklearn(features, k_nn)
        mean_distance = distances.mean(axis=1)
        
        return 1.0 / (mean_distance + 1e-5)

    def _build_cluster_df(self, cluster_labels, existing_indices):
        """Build and sort cluster information."""
        import pandas as pd
        
        labels = np.copy(cluster_labels)
        
        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
        
        if len(cluster_ids) > 0:
            counts = np.bincount(labels[existing_indices], minlength=cluster_ids.max() + 1)
            cluster_labeled_counts = counts[cluster_ids]
        else:
            cluster_labeled_counts = np.array([], dtype=int)
        
        clusters_df = pd.DataFrame({
            'cluster_id': cluster_ids,
            'cluster_size': cluster_sizes,
            'existing_count': cluster_labeled_counts,
            'neg_cluster_size': -1 * cluster_sizes
        })
        
        clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]
        clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
        
        labels[existing_indices] = -1
        
        return clusters_df['cluster_id'].values, labels

    def select_samples_typiclust(self, features, cluster_labels, existing_indices, budget):
        """Select samples using TypiClust strategy."""
        sorted_clusters, labels = self._build_cluster_df(
            cluster_labels, existing_indices
        )
        
        if len(sorted_clusters) == 0:
            print("Fallback: No valid clusters found. Selecting randomly.")
            unlabeled_mask = np.ones(len(features), dtype=bool)
            unlabeled_mask[existing_indices] = False
            unlabeled_indices = np.where(unlabeled_mask)[0]
            return np.random.choice(unlabeled_indices, size=min(budget, len(unlabeled_indices)), replace=False)
        
        selected = []
        for i in range(budget):
            cluster = sorted_clusters[i % len(sorted_clusters)]
            indices = np.where(labels == cluster)[0]
            
            if len(indices) == 0:
                for j in range(len(sorted_clusters)):
                    alt_cluster = sorted_clusters[(i + j) % len(sorted_clusters)]
                    indices = np.where(labels == alt_cluster)[0]
                    if len(indices) > 0:
                        cluster = alt_cluster
                        break
                if len(indices) == 0:
                    break
            
            rel_feats = features[indices]
            k_nn = max(1, min(self.K_NN, len(indices) // 2))
            
            cluster_typicalities = self.compute_typicality(rel_feats, k_nn)
            best_local_idx = cluster_typicalities.argmax()
            idx = indices[best_local_idx]
            
            selected.append(idx)
            labels[idx] = -1
        
        return np.array(selected, dtype=int)

    # --- CoreSetProb Methods ---

    def probabilistic_furthest_first(self, unlabeled_features, labeled_features, n, random_state):
        """Probabilistic k-means++ style sampling for k-center."""
        m = unlabeled_features.shape[0]
        
        n_local_trials = self.n_local_trials
        if n_local_trials is None:
            n_local_trials = max(2 + int(np.log(n)), 1)
        
        unlabeled_squared_norms = (unlabeled_features ** 2).sum(axis=1)
        
        if labeled_features.shape[0] == 0:
            first_idx = random_state.randint(m)
            selected_idxs = [first_idx]
            closest_dist_sq = euclidean_distances(
                unlabeled_features[[first_idx], :],
                unlabeled_features,
                Y_norm_squared=unlabeled_squared_norms,
                squared=True
            )[0]
            start_idx = 1
        else:
            labeled_squared_norms = (labeled_features ** 2).sum(axis=1)
            dist_to_labeled = euclidean_distances(
                labeled_features,
                unlabeled_features,
                X_norm_squared=labeled_squared_norms,
                Y_norm_squared=unlabeled_squared_norms,
                squared=True
            )
            closest_dist_sq = np.amin(dist_to_labeled, axis=0)
            selected_idxs = []
            start_idx = 0
        
        for i in range(start_idx, n):
            current_pot = closest_dist_sq.sum()
            
            if current_pot == 0:
                remaining_m_indices = np.setdiff1d(np.arange(m), selected_idxs)
                if remaining_m_indices.size > 0:
                    idx = random_state.choice(remaining_m_indices)
                    selected_idxs.append(idx)
                continue
            
            rand_vals = random_state.random_sample(n_local_trials) * current_pot
            candidate_idxs = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
            np.clip(candidate_idxs, 0, m - 1, out=candidate_idxs)
            
            candidates_features = unlabeled_features[candidate_idxs]
            distance_to_candidates = euclidean_distances(
                candidates_features,
                unlabeled_features,
                Y_norm_squared=unlabeled_squared_norms,
                squared=True
            )
            
            updated_dist = np.minimum(closest_dist_sq, distance_to_candidates)
            candidates_pot = updated_dist.sum(axis=1)
            
            best_candidate_idx = np.argmin(candidates_pot)
            best_idx = candidate_idxs[best_candidate_idx]
            
            selected_idxs.append(best_idx)
            closest_dist_sq = updated_dist[best_candidate_idx]
            
        return np.array(selected_idxs, dtype=int)

    # --- Main AL Loop ---

    def active_learn_sampler(self, run, task_stream, task_i):
        """Execute TypiKmeans active learning strategy."""
        accuracies = np.array([])
        task_buffer = np.array([], dtype=int)
        
        for alc in range(self.al_budget):
            print(f'AL cycle: {alc + 1} / {self.al_budget}')
            random_state = np.random.RandomState(self.random_state + alc)

            x_train, _ = self.current_task[0]
            all_features, _ = self.extract_features_and_outputs(x_train)
            
            if alc < 2:
                # --- TypiClust Phase ---
                print("Strategy: TypiClust (Exploration)")
                n_clusters = min(len(self.idx_labeled) + self.n_samples_per_al_cycle, self.MAX_NUM_CLUSTERS)
                print(f"Clustering into {n_clusters} clusters")
                cluster_labels = self.get_clusters(all_features, n_clusters)
                
                selected_idxs = self.select_samples_typiclust(
                    all_features, 
                    cluster_labels, 
                    self.idx_labeled.astype(int), 
                    self.n_samples_per_al_cycle
                )
            else:
                # --- CoreSetProb Phase ---
                print("Strategy: CoreSetProb (Exploitation/Coverage)")
                unlabeled_features = all_features[self.idx_unlabeled]
                
                if self.idx_labeled.size > 0:
                    labeled_features = all_features[self.idx_labeled]
                else:
                    labeled_features = np.empty((0, all_features.shape[1]))
                
                print(f"Selecting from {len(self.idx_unlabeled)} unlabeled samples")
                
                local_indices = self.probabilistic_furthest_first(
                    unlabeled_features,
                    labeled_features,
                    self.n_samples_per_al_cycle,
                    random_state
                )
                selected_idxs = self.idx_unlabeled[local_indices]

            print(f"Selected {len(selected_idxs)} samples")

            # Update labeled/unlabeled sets
            self.idx_labeled = np.concatenate([self.idx_labeled, selected_idxs])
            self.idx_unlabeled = np.setdiff1d(
                self.idx_unlabeled, 
                selected_idxs, 
                assume_unique=True
            )
            task_buffer = np.concatenate([task_buffer, selected_idxs])

            # Train and evaluate
            self.agent.learn_task(self.current_task, task_buffer, alc == 0)
            accuracies = self.agent.evaluate(task_stream, alc, self.al_budget)
            self.save_acc_to_csv(accuracies, run, task_i, alc)

        return accuracies
