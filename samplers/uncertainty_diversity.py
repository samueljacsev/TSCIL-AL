from sklearn.cluster import KMeans
from types import SimpleNamespace
from agents.base import BaseLearner
from samplers.base import BaseSampler
from utils.data import Dataloader_from_numpy
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

    def compute_uncertainty(self, outputs, metric='least_confidence'):
        """
        Compute uncertainty for a batch of outputs.

        Args:
            outputs: Model outputs (logits or probabilities).
            metric: Uncertainty metric ('entropy', 'margin', 'least_confidence').

        Returns:
            uncertainties: Array of uncertainty scores.
        """
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

        if metric == 'entropy':
            uncertainties = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        elif metric == 'margin':
            sorted_probs = -np.sort(-probabilities, axis=1)  # Sort in descending order
            uncertainties = 1 - (sorted_probs[:, 0] - sorted_probs[:, 1])
        elif metric == 'least_confidence':
            uncertainties = 1 - np.max(probabilities, axis=1)
        else:
            raise ValueError(f"Unknown uncertainty metric: {metric}")

        return uncertainties

    def active_learn_task(self, run, task_stream, task_i, metric='least_confidence'):
        """
        Selects the next few samples to be labeled based on uncertainty sampling with diversity.

        Args:
            task_stream: Task stream containing tasks.
            task_i: Index of the current task.
            metric: Uncertainty metric ('entropy', 'margin', 'least_confidence').
        """
        
        # Set random seeds for reproducibility
        np.random.seed(run)
        torch.manual_seed(run)
        torch.cuda.manual_seed_all(run)
        
        if self.args.uncertainty_type is not None:
            metric = self.args.uncertainty_type

        task = task_stream.tasks[task_i]
        (x_train, y_train) = task[0]  # y_train is not used for unlabeled data

        n_samples_current_task = x_train.shape[0]
        print('Number of samples in current task:', n_samples_current_task)

        n_samples_per_al_cycle = self.get_n_samples_per_al_cycle(n_samples_current_task)
        n_clusters = n_samples_per_al_cycle # Number of clusters for diversity sampling

        # Initialize unlabeled indices
        idx_unlabeled = np.arange(n_samples_current_task)

        for alc in range(self.al_budget):
            print(f'Run: {run}, Task: {task_i}, AL cycle: {alc + 1} / {self.al_budget}')

            if alc == 0:
                # Randomly select the first batch of samples
                np.random.seed(run)
                np.random.shuffle(idx_unlabeled)
                selected_idxs = idx_unlabeled[:n_samples_per_al_cycle]
            else:
                # Step 1: Extract embeddings for all unlabeled samples
                eval_dataloader = Dataloader_from_numpy(
                    x_train[idx_unlabeled],
                    np.zeros(len(idx_unlabeled)),  # Dummy labels
                    self.batch_size,
                    shuffle=False
                )

                all_features = []
                all_outputs = []
                for batch_id, (batch_x, _) in enumerate(eval_dataloader):
                    batch_x = batch_x.to(self.agent.device)
                    with torch.no_grad():
                        # Extract embeddings and outputs
                        features = self.agent.model.feature(batch_x)  # Feature extraction
                        outputs = self.agent.model(batch_x)  # Forward pass
                    all_features.append(features.cpu().numpy())
                    all_outputs.append(outputs)
                all_features = np.vstack(all_features)
                all_outputs = torch.cat(all_outputs, dim=0)

                # Step 2: Cluster the embeddings
                kmeans = KMeans(n_clusters=n_clusters, random_state=run)
                cluster_labels = kmeans.fit_predict(all_features)

                # Step 3: Compute uncertainty scores
                uncertainties = self.compute_uncertainty(all_outputs, metric=metric)

                # Step 4: Select the most uncertain sample from each cluster
                selected_idxs = []
                for cluster in range(n_clusters):
                    cluster_indices = np.where(cluster_labels == cluster)[0]  # Indices in this cluster
                    cluster_uncertainties = uncertainties[cluster_indices]  # Uncertainty scores for this cluster
                    slct_idx = cluster_indices[np.argsort(-cluster_uncertainties)[0]]
                    # Add the selected indices to the list
                    selected_idxs.append(idx_unlabeled[slct_idx])
                    
                selected_idxs = np.array(selected_idxs[:n_samples_per_al_cycle])
                
                
            # Update unlabeled indices
            idx_unlabeled = np.setdiff1d(idx_unlabeled, selected_idxs)

            new_task = (alc == 0)  # First cycle is a new task
            # Train the agent on the newly labeled data
            self.agent.learn_task(task, selected_idxs, new_task)
            accuracies = self.agent.evaluate(task_stream, alc, self.al_budget)
            self.save_acc_to_csv(accuracies, run, task_i, alc, f'_{metric}')