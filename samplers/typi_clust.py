# -*- coding: UTF-8 -*-
from types import SimpleNamespace
from agents.base import BaseLearner
from samplers.base import BaseSampler
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np

# Try to import FAISS for GPU-accelerated NN search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class TypiClustSampler(BaseSampler):
    """
    TypiClustSampler: A sampler implementing the TypiClust strategy.
    
    Based on the original TypiClust paper implementation:
    - Clusters all samples (labeled + unlabeled)
    - Prioritizes clusters with fewer labeled samples and larger size
    - Selects the most typical (highest density) sample from each cluster
    - Uses FAISS for fast GPU-accelerated nearest neighbor search when available
    """
    
    MIN_CLUSTER_SIZE = 5
    MAX_NUM_CLUSTERS = 500
    K_NN = 20  # Default number of neighbors for typicality

    def __init__(self,
                 agent: BaseLearner,
                 exp_args: SimpleNamespace,
                 args: SimpleNamespace):
        super().__init__(agent, exp_args, args, name='TypiClust')

    def get_clusters(self, features, n_clusters):
        """
        Perform K-means clustering on features for diversity.
        Uses MiniBatchKMeans for large datasets or many clusters for efficiency.
        
        Args:
            features: Feature vectors to cluster.
            n_clusters: Number of clusters to create.
            
        Returns:
            cluster_labels: Array of cluster assignments.
        """
        n_clusters = min(n_clusters, len(features), self.MAX_NUM_CLUSTERS)
        
        # Use MiniBatchKMeans for better scalability (matches original implementation)
        if n_clusters > 50:
            print(f"Using MiniBatchKMeans for {n_clusters} clusters on {len(features)} samples")
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, 
                                    batch_size=min(5000, len(features)),
                                    random_state=self.random_state)
        else:
            print(f"Using KMeans for {n_clusters} clusters on {len(features)} samples")
            kmeans = KMeans(n_clusters=n_clusters, 
                          random_state=self.random_state)
        
        return kmeans.fit_predict(features)

    def _get_nn_faiss(self, features, num_neighbors):
        """
        Calculate nearest neighbors using FAISS (GPU-accelerated when available).
        
        Args:
            features: Feature vectors (N x D).
            num_neighbors: Number of neighbors to find.
            
        Returns:
            distances: Distance to each neighbor (N x num_neighbors).
            indices: Indices of neighbors (N x num_neighbors).
        """
        d = features.shape[1]
        features = np.ascontiguousarray(features.astype(np.float32))
        
        cpu_index = faiss.IndexFlatL2(d)
        
        # Try to use GPU if available
        try:
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
            index = gpu_index
        except Exception:
            index = cpu_index
        
        index.add(features)
        distances, indices = index.search(features, num_neighbors + 1)
        
        # Index 0 is the same sample (distance 0), drop it
        return distances[:, 1:], indices[:, 1:]

    def _get_nn_sklearn(self, features, num_neighbors):
        """
        Calculate nearest neighbors using sklearn (CPU fallback).
        Memory-efficient version that doesn't compute full distance matrix.
        
        Args:
            features: Feature vectors (N x D).
            num_neighbors: Number of neighbors to find.
            
        Returns:
            distances: Distance to each neighbor (N x num_neighbors).
            indices: Indices of neighbors (N x num_neighbors).
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
        
        Args:
            features: Feature vectors to compute typicality for.
            k_nn: Number of neighbors to use. Defaults to min(K_NN, len(features)//2).
            
        Returns:
            typicalities: Array of typicality scores.
        """
        if k_nn is None:
            k_nn = min(self.K_NN, len(features) // 2)
        
        # Ensure we have at least 1 neighbor
        k_nn = max(1, min(k_nn, len(features) - 1))
        
        if FAISS_AVAILABLE:
            distances, _ = self._get_nn_faiss(features, k_nn)
        else:
            distances, _ = self._get_nn_sklearn(features, k_nn)
        
        mean_distance = distances.mean(axis=1)
        
        # Typicality is inverse of mean distance (high density = high typicality)
        typicalities = 1.0 / (mean_distance + 1e-5)
        return typicalities

    def _build_cluster_df(self, cluster_labels, relevant_indices, existing_indices):
        """
        Build cluster information similar to original implementation using pandas.
        
        Args:
            cluster_labels: Cluster assignment for each sample in relevant_indices.
            relevant_indices: Combined labeled + unlabeled indices.
            existing_indices: Indices within relevant_indices that are already labeled.
            
        Returns:
            sorted_clusters: Cluster IDs sorted by priority (least labeled, largest size).
            labels: Mutable copy of cluster_labels with labeled samples marked as -1.
        """
        import pandas as pd
        
        labels = np.copy(cluster_labels)
        
        # Count cluster sizes and labeled samples per cluster
        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
        cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(cluster_ids))
        
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

    def select_samples(self, features, cluster_labels, existing_indices, budget):
        """
        Select samples using TypiClust strategy (matches original implementation).
        
        Args:
            features: Feature vectors for all relevant samples.
            cluster_labels: Cluster assignments for all relevant samples.
            existing_indices: Indices of already labeled samples (within this array).
            budget: Number of samples to select.
            
        Returns:
            selected: Indices of selected samples (within the features array).
        """
        sorted_clusters, labels = self._build_cluster_df(
            cluster_labels, np.arange(len(features)), existing_indices
        )
        
        if len(sorted_clusters) == 0:
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
            
            # Compute typicality within this cluster
            rel_feats = features[indices]
            k_nn = min(self.K_NN, len(indices) // 2)
            typicality = self.compute_typicality(rel_feats, k_nn)
            
            # Select the most typical sample
            best_local_idx = typicality.argmax()
            idx = indices[best_local_idx]
            
            selected.append(idx)
            # Mark as selected so it won't be chosen again
            labels[idx] = -1
        
        return np.array(selected, dtype=int)

    def active_learn_sampler(self, run, task_stream, task_i):
        """Execute TypiClust active learning strategy."""
        accuracies = np.array([])
        task_buffer = np.array([], dtype=int)
        
        for alc in range(self.al_budget):
            print(f'AL cycle: {alc + 1} / {self.al_budget}')

            # Extract features from training data
            x_train, _ = self.current_task[0]
            all_features, _ = self.extract_features_and_outputs(x_train)
            
            # Determine number of clusters (matches original implementation)
            n_clusters = min(len(self.idx_labeled) + self.n_samples_per_al_cycle, self.MAX_NUM_CLUSTERS)
            print(f"Step 1: Clustering into {n_clusters} clusters for diversity")
            cluster_labels = self.get_clusters(all_features, n_clusters)
            
            # Build relevant indices (labeled + unlabeled for this task)
            relevant_indices = np.concatenate([self.idx_labeled, self.idx_unlabeled]).astype(int)
            relevant_features = all_features[relevant_indices]
            relevant_clusters = cluster_labels[relevant_indices]
            
            # existing_indices: positions within relevant_indices that are labeled
            existing_indices = np.arange(len(self.idx_labeled))
            
            # Select samples using TypiClust strategy
            print("Step 2: Selecting typical samples from clusters")
            selected_local = self.select_samples(
                relevant_features, 
                relevant_clusters, 
                existing_indices, 
                self.n_samples_per_al_cycle
            )
            
            # Map back to original indices
            selected_idxs = relevant_indices[selected_local]
            
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