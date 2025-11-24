from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances
from types import SimpleNamespace
from agents.base import BaseLearner
from samplers.base import BaseSampler
import numpy as np
import torch


class TypiClustUncertaintyDiversitySampler(BaseSampler):
    """
    TypiClustUncertaintyDiversitySampler: A hybrid sampler combining multiple strategies.
    - Cycles [0:1]: TypiClust (typicality-based selection with clustering)
    - Cycles [2:-2]: Uncertainty-Diversity (cluster-based uncertainty sampling)
    - Cycles [-2:]: Pure Uncertainty (no clustering/diversity)
    """

    def __init__(self,
                 agent: BaseLearner,
                 exp_args: SimpleNamespace,
                 args: SimpleNamespace):
        super().__init__(agent, exp_args, args, name='TypiClustUncertaintyDiversity2')
        self.metric = args.uncertainty_type if args.uncertainty_type else 'least_confidence'

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
        cluster_order = np.lexsort((labeled_counts, -cluster_sizes,))
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

    def compute_uncertainty(self, outputs):
        """
        Compute uncertainty for a batch of outputs.

        Args:
            outputs: Model outputs (logits or probabilities).

        Returns:
            uncertainties: Array of uncertainty scores.
        """
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        if self.metric == 'entropy':
            uncertainties = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        elif self.metric == 'margin':
            sorted_probs = -np.sort(-probabilities, axis=1)  # Sort in descending order
            uncertainties = 1 - (sorted_probs[:, 0] - sorted_probs[:, 1])
        elif self.metric == 'least_confidence':
            uncertainties = 1 - np.max(probabilities, axis=1)
        else:
            raise ValueError(f"Unknown uncertainty metric: {self.metric}")

        return uncertainties

    def active_learn_sampler(self, run, task_stream, task_i):
        """Execute hybrid TypiClust-Uncertainty-Diversity active learning strategy."""
        accuracies = np.array([])
        task_buffer = np.array([], dtype=int)
        
        for alc in range(self.al_budget):
            print(f'AL cycle: {alc + 1} / {self.al_budget}')

            if alc < 1:
                # First cycles: Use TypiClust strategy with clustering
                print("Using TypiClust strategy for initial selection")
                x_train, _ = self.current_task[0]
                all_features, _ = self.extract_features_and_outputs(x_train)
                
                # Cluster all samples
                n_clusters = min(self.n_samples_per_al_cycle, self.idx_unlabeled.size)
                print(f"Step 1: Clustering for Diversity - {n_clusters} clusters")
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
            elif alc < self.al_budget - 2:
                # Cycles 2-3: Use Uncertainty-Diversity strategy (with clustering)
                print("Using Uncertainty-Diversity strategy (with clustering)")
                # Extract features and outputs for unlabeled samples
                x_train, _ = self.current_task[0]
                x_unlabeled = x_train[self.idx_unlabeled]
                unlabeled_features, unlabeled_outputs = self.extract_features_and_outputs(x_unlabeled)

                # Cluster the embeddings for diversity
                # Number of clusters = number of labeled samples + samples to select this cycle
                n_clusters = min(len(self.idx_labeled) + self.n_samples_per_al_cycle, self.idx_unlabeled.size)
                print(f"Clustering into {n_clusters} clusters for diversity")
                cluster_labels = self.get_clusters(unlabeled_features, n_clusters)

                # Compute uncertainty scores
                print(f"Computing uncertainty using {self.metric} metric")
                uncertainties = self.compute_uncertainty(unlabeled_outputs)

                # Calculate mean uncertainty for each cluster using vectorized operations
                cluster_sums = np.bincount(cluster_labels, weights=uncertainties)
                cluster_counts = np.bincount(cluster_labels)
                # Only keep clusters with at least one sample
                valid_clusters = cluster_counts > 20
                cluster_mean_uncertainties = np.divide(cluster_sums[valid_clusters], 
                                                       cluster_counts[valid_clusters])
                cluster_info = np.where(valid_clusters)[0]
                
                # Sort clusters by mean uncertainty (descending)
                sorted_cluster_indices = np.argsort(-cluster_mean_uncertainties)[:self.n_samples_per_al_cycle]
                top_uncertain_clusters = cluster_info[sorted_cluster_indices]
                
                # Select the most uncertain exemplar from each of the top clusters
                selected_local_idxs = []
                for cluster in top_uncertain_clusters:
                    cluster_indices = np.where(cluster_labels == cluster)[0]
                    cluster_uncertainties = uncertainties[cluster_indices]
                    most_uncertain_idx = cluster_indices[np.argmax(cluster_uncertainties)]
                    selected_local_idxs.append(most_uncertain_idx)
                
                selected_local_idxs = np.array(selected_local_idxs, dtype=int)
                selected_idxs = self.idx_unlabeled[selected_local_idxs]
            else:
                # Cycles 4+: Use pure Uncertainty sampling (no clustering/diversity)
                print("Using pure Uncertainty sampling (no diversity)")
                # Extract outputs for unlabeled samples
                x_train, _ = self.current_task[0]
                x_unlabeled = x_train[self.idx_unlabeled]
                _, unlabeled_outputs = self.extract_features_and_outputs(x_unlabeled)

                # Compute uncertainty scores
                print(f"Computing uncertainty using {self.metric} metric")
                uncertainties = self.compute_uncertainty(unlabeled_outputs)

                # Select the most uncertain samples
                local_indices = np.argsort(-uncertainties)[:self.n_samples_per_al_cycle]
                selected_idxs = self.idx_unlabeled[local_indices]

            # Update labeled and unlabeled sets
            self.idx_labeled = np.concatenate([self.idx_labeled, selected_idxs])
            self.idx_unlabeled = np.setdiff1d(self.idx_unlabeled, selected_idxs, 
                                             assume_unique=True)
            
            # Add selected samples to the task buffer
            task_buffer = np.concatenate([task_buffer, selected_idxs])

            # Train and evaluate
            self.agent.learn_task(self.current_task, task_buffer, alc == 0)
            accuracies = self.agent.evaluate(task_stream, alc, self.al_budget)
            self.save_acc_to_csv(accuracies, run, task_i, alc, f'_{self.metric}')

        return accuracies