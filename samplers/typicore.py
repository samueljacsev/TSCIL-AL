# -*- coding: UTF-8 -*-
from types import SimpleNamespace
from agents.base import BaseLearner
from samplers.base import BaseSampler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances
import numpy as np


class TypiCoreSampler(BaseSampler):
    """
    TypiCoreSampler: A hybrid sampler combining TypiClust and CoreSet strategies.
    
    Strategy:
    - First 2 AL cycles: Use TypiClust (clustering + typicality selection)
    - Subsequent cycles: Use CoreSet (greedy furthest-first selection)
    
    This combines the benefits of both approaches:
    - TypiClust provides good initial coverage with typical samples
    - CoreSet ensures diverse coverage by selecting furthest points
    """
    
    MIN_CLUSTER_SIZE = 3
    MAX_NUM_CLUSTERS = np.inf
    K_NN = 20

    def __init__(self,
                 agent: BaseLearner,
                 exp_args: SimpleNamespace,
                 args: SimpleNamespace):
        super().__init__(agent, exp_args, args, name='TypiCore')
        print("TypiCore initialized: First 2 cycles use TypiClust, then CoreSet")

    # ==================== TypiClust Methods ====================
    
    def get_clusters(self, features, n_clusters):
        """
        Perform K-means clustering on features for diversity.
        Uses MiniBatchKMeans for large datasets or many clusters for efficiency.
        """
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
        """
        Calculate nearest neighbors using sklearn.
        Memory-efficient version that doesn't compute full distance matrix.
        """
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=num_neighbors + 1, algorithm='auto', metric='euclidean')
        nn.fit(features)
        distances, indices = nn.kneighbors(features)
        
        # Index 0 is the same sample, drop it
        return distances[:, 1:], indices[:, 1:]

    def compute_typicality(self, features, k_nn=None):
        """
        Compute typicality scores based on k-nearest neighbor distances.
        High typicality = low mean distance to neighbors = high density region.
        """
        if k_nn is None:
            k_nn = min(self.K_NN, len(features) // 2)
        
        # Ensure we have at least 1 neighbor
        k_nn = max(1, min(k_nn, len(features) - 1))
        
        distances, _ = self._get_nn_sklearn(features, k_nn)
        mean_distance = distances.mean(axis=1)
        
        # Typicality is inverse of mean distance (high density = high typicality)
        typicalities = 1.0 / (mean_distance + 1e-5)
        return typicalities

    def _build_cluster_df(self, cluster_labels, existing_indices):
        """
        Build cluster information similar to original implementation using pandas.
        """
        import pandas as pd
        
        labels = np.copy(cluster_labels)
        
        # Count cluster sizes and labeled samples per cluster
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
        
        # Drop clusters that are too small
        clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]
        
        # Sort: lowest existing count first, then largest cluster size
        clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
        
        # Mark existing labeled samples as -1 so they won't be selected
        labels[existing_indices] = -1
        
        return clusters_df['cluster_id'].values, labels

    def select_samples_typiclust(self, features, cluster_labels, existing_indices, budget):
        """
        Select samples using TypiClust strategy.
        """
        sorted_clusters, labels = self._build_cluster_df(
            cluster_labels, existing_indices
        )
        
        if len(sorted_clusters) == 0:
            print("Fallback: No valid clusters found for selection.")
            # Fallback: no valid clusters, select randomly from unlabeled
            unlabeled_mask = np.ones(len(features), dtype=bool)
            unlabeled_mask[existing_indices] = False
            unlabeled_indices = np.where(unlabeled_mask)[0]
            return np.random.choice(unlabeled_indices, size=min(budget, len(unlabeled_indices)), replace=False)
        
        selected = []

        for i in range(budget):
            # Round-robin through clusters
            cluster = sorted_clusters[i % len(sorted_clusters)]
            
            # Get indices of samples in this cluster (not yet selected)
            indices = np.where(labels == cluster)[0]
            
            if len(indices) == 0:
                # This cluster is exhausted, find next available cluster
                for j in range(len(sorted_clusters)):
                    alt_cluster = sorted_clusters[(i + j) % len(sorted_clusters)]
                    indices = np.where(labels == alt_cluster)[0]
                    if len(indices) > 0:
                        cluster = alt_cluster
                        break
                
                if len(indices) == 0:
                    # All clusters exhausted
                    break
            
            # Compute typicality for this cluster with adaptive K_NN
            rel_feats = features[indices]
            k_nn = min(self.K_NN, len(indices) // 2)
            k_nn = max(1, k_nn)
            
            cluster_typicalities = self.compute_typicality(rel_feats, k_nn)
            best_local_idx = cluster_typicalities.argmax()
            idx = indices[best_local_idx]
            
            selected.append(idx)
            # Mark as selected so it won't be chosen again
            labels[idx] = -1
        
        return np.array(selected, dtype=int)

    # ==================== CoreSet Methods ====================
    
    def furthest_first(self, unlabeled_features, labeled_features, n):
        """
        Greedy furthest-first traversal for k-center problem.
        
        Iteratively selects n samples from unlabeled_features that are
        furthest from the labeled set, maximizing minimum distance to
        already selected points.
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

    def select_samples_coreset(self, all_features):
        """
        Select samples using CoreSet strategy.
        """
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
        return selected_idxs

    # ==================== Main Active Learning Loop ====================
    
    def active_learn_sampler(self, run, task_stream, task_i):
        """Execute TypiCore active learning strategy."""
        accuracies = np.array([])
        task_buffer = np.array([], dtype=int)
        
        for alc in range(self.al_budget):
            print(f'AL cycle: {alc + 1} / {self.al_budget}')

            # Extract features from training data
            x_train, y_train = self.current_task[0]
            all_features, _ = self.extract_features_and_outputs(x_train)
            
            # Switch strategy based on cycle number
            if alc % 2 == 0:
                # First 2 cycles: Use TypiClust
                print(f"Using TypiClust strategy (cycle {alc + 1}/2)")
                
                # Determine number of clusters
                n_clusters = min(len(self.idx_labeled) + self.n_samples_per_al_cycle, self.MAX_NUM_CLUSTERS)
                print(f"Step 1: Clustering into {n_clusters} clusters for diversity")
                cluster_labels = self.get_clusters(all_features, n_clusters)
                
                # Select samples using TypiClust strategy
                print("Step 2: Selecting typical samples from clusters")
                selected_idxs = self.select_samples_typiclust(
                    all_features, 
                    cluster_labels, 
                    self.idx_labeled.astype(int), 
                    self.n_samples_per_al_cycle
                )
                
                # Safety check for class diversity
                selected_labels = np.array([y_train[idx] for idx in selected_idxs])
                if len(np.unique(selected_labels)) < 2:
                    print("Warning: Selected samples contain less than 2 unique classes.")
                    # Replace the last selected sample with a random unlabeled sample of a different class
                    unlabeled_mask = np.ones(len(all_features), dtype=bool)
                    unlabeled_mask[self.idx_labeled.astype(int)] = False
                    unlabeled_indices = np.where(unlabeled_mask)[0]
                    for alt_idx in unlabeled_indices:
                        if y_train[alt_idx] not in selected_labels:
                            print(f"Replacing index {selected_idxs[-1]} with index {alt_idx} of class {y_train[alt_idx]}")
                            selected_idxs[-1] = alt_idx
                            break
            else:
                # Subsequent cycles: Use CoreSet
                print(f"Using CoreSet strategy (cycle {alc + 1})")
                selected_idxs = self.select_samples_coreset(all_features)
            
            # Update labeled and unlabeled sets
            self.idx_labeled = np.concatenate([self.idx_labeled, selected_idxs])
            self.idx_unlabeled = np.setdiff1d(self.idx_unlabeled, selected_idxs, 
                                             assume_unique=True)

            # Add selected samples to the task buffer
            task_buffer = np.concatenate([task_buffer, selected_idxs])

            print(f"Selected {len(selected_idxs)} samples. Total labeled: {len(self.idx_labeled)}, Remaining unlabeled: {len(self.idx_unlabeled)}")

            # Train and evaluate
            self.agent.learn_task(self.current_task, task_buffer, alc == 0)
            accuracies = self.agent.evaluate(task_stream, alc, self.al_budget)
            self.save_acc_to_csv(accuracies, run, task_i, alc)

        return accuracies
