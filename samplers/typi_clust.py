# -*- coding: UTF-8 -*-
from types import SimpleNamespace
from agents.base import BaseLearner
from samplers.base import BaseSampler
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np


class TypiClustSampler(BaseSampler):
    """
    TypiClustSampler: A sampler implementing the TypiClust strategy for 2-class tasks.
    """

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
        n_clusters = min(n_clusters, len(self.idx_unlabeled))
        
        # Use MiniBatchKMeans for better scalability with large datasets or many clusters
        if n_clusters > 50 or len(features) > 10000:
            print(f"Using MiniBatchKMeans for {n_clusters} clusters on {len(features)} samples")
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, 
                                    batch_size=min(5000, len(features)),
                                    random_state=self.random_state)
        else:
            print(f"Using KMeans for {n_clusters} clusters on {len(features)} samples")
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        
        return kmeans.fit_predict(features)


    def compute_typicality(self, features, k_sym):
        """Compute typicality scores based on k-nearest neighbor distances."""
        from sklearn.metrics.pairwise import euclidean_distances
        
        distances = euclidean_distances(features, features)
        # Exclude self-distance (first column after sorting)
        k_nearest_distances = np.sort(distances, axis=1)[:, 1:k_sym + 1]
        mean_distances = np.mean(k_nearest_distances, axis=1)
        
        # Typicality is inverse of mean distance
        typicalities = 1.0 / (mean_distances + 1e-10)
        return typicalities

    def _cluster_priority(self, cluster_labels):
        """Rank clusters by labeled coverage (ascending) and size (descending)."""
        unlabeled_assignments = cluster_labels[self.idx_unlabeled]
        unique_clusters, cluster_sizes = np.unique(unlabeled_assignments, return_counts=True)
        
        # Count labeled samples per cluster
        if self.idx_labeled.size > 0:
            labeled_assignments = cluster_labels[self.idx_labeled]
            labeled_counts = np.array([np.sum(labeled_assignments == c) for c in unique_clusters])
        else:
            labeled_counts = np.zeros(len(unique_clusters), dtype=int)
        
        # Sort by labeled count (ascending), then by cluster size (descending)
        cluster_order = np.lexsort((-cluster_sizes, labeled_counts))
        return unique_clusters[cluster_order], unlabeled_assignments

    def _select_typical_indices(self, sorted_clusters, unlabeled_assignments, typicalities, target):
        """Select most typical samples from clusters in round-robin fashion."""
        # Map each cluster to its unlabeled sample indices
        cluster_indices = {
            cluster: list(np.where(unlabeled_assignments == cluster)[0]) 
            for cluster in sorted_clusters
        }
        
        selected = []
        while len(selected) < target:
            selected_this_round = False
            
            for cluster in sorted_clusters:
                if len(selected) >= target:
                    break
                    
                candidates = cluster_indices[cluster]
                if not candidates:
                    continue
                
                # Select sample with highest typicality score
                best_idx = max(candidates, key=lambda idx: typicalities[idx])
                selected.append(best_idx)
                candidates.remove(best_idx)
                selected_this_round = True
            
            # Stop if no cluster had candidates
            if not selected_this_round:
                break
        
        return np.array(selected, dtype=int)

    def active_learn_sampler(self, run, task_stream, task_i):
        """Execute TypiClust active learning strategy."""
        accuracies = np.array([])
        
        for alc in range(self.al_budget):
            print(f'AL cycle: {alc + 1} / {self.al_budget}')

            # Extract features from training data
            x_train, _ = self.current_task[0]
            all_features, _ = self.extract_features_and_outputs(x_train)
            
            # Cluster all samples
            print("Step 1: Clustering for Diversity")
            n_clusters = min(len(self.idx_labeled) + self.n_samples_per_al_cycle, 
                           self.idx_unlabeled.size)
            cluster_labels = self.get_clusters(all_features, n_clusters)
            sorted_clusters, unlabeled_assignments = self._cluster_priority(cluster_labels)
            
            # Compute typicality scores for unlabeled samples
            print("Step 2: Querying Typical Examples")
            unlabeled_features = all_features[self.idx_unlabeled]
            k_nn = min(max(self.n_samples_per_al_cycle, 20), len(unlabeled_features) - 1)
            typicalities = self.compute_typicality(unlabeled_features, k_nn)
            
            # Select most typical samples from each cluster
            local_indices = self._select_typical_indices(
                sorted_clusters, unlabeled_assignments, typicalities, 
                self.n_samples_per_al_cycle
            )
            selected_idxs = self.idx_unlabeled[local_indices]
            
            # Update labeled and unlabeled sets
            self.idx_labeled = np.concatenate([self.idx_labeled, selected_idxs])
            self.idx_unlabeled = np.setdiff1d(self.idx_unlabeled, selected_idxs, 
                                             assume_unique=True)

            # Train and evaluate
            self.agent.learn_task(self.current_task, selected_idxs, alc == 0)
            accuracies = self.agent.evaluate(task_stream, alc, self.al_budget)
            self.save_acc_to_csv(accuracies, run, task_i, alc)

        return accuracies