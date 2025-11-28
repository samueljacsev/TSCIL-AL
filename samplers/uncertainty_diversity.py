from sklearn.cluster import KMeans, MiniBatchKMeans
from types import SimpleNamespace
from agents.base import BaseLearner
from samplers.base import BaseSampler
import numpy as np
import torch


class UncertaintyDiversitySampler(BaseSampler):
    """
    UncertaintyDiversitySampler: A sampler with uncertainty sampling and diversity enforcement.
    """

    def __init__(self,
                 agent: BaseLearner,
                 exp_args: SimpleNamespace,
                 args: SimpleNamespace):
        super().__init__(agent, exp_args, args, name='UncertaintyWithDiversity')
        self.metric = args.uncertainty_type if args.uncertainty_type else 'entropy'

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
        elif self.metric == 'lc':
            uncertainties = 1 - np.max(probabilities, axis=1)
        else:
            raise ValueError(f"Unknown uncertainty metric: {self.metric}")

        return uncertainties

    def active_learn_sampler(self, run, task_stream, task_i):
        """Execute uncertainty-based active learning with diversity enforcement."""
        accuracies = np.array([])
        task_buffer = np.array([], dtype=int)
        
        for alc in range(self.al_budget):
            print(f'AL cycle: {alc + 1} / {self.al_budget}')

            if alc == 0:
                # Randomly select the first batch of samples
                np.random.shuffle(self.idx_unlabeled)
                selected_idxs = self.idx_unlabeled[:self.n_samples_per_al_cycle].copy()
            else:
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
                valid_clusters = cluster_counts > 0
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
                
                selected_local_idxs = np.array(selected_local_idxs)
                selected_idxs = self.idx_unlabeled[selected_local_idxs]

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