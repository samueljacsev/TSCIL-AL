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



    def active_learn_task(self, run, task_stream, task_i, metric='least_confidence', classes_in_each_task=None):
        """
        Selects the next few samples to be labeled based on uncertainty sampling with diversity.

        Args:
            run: Run identifier.
            task_stream: Task stream containing tasks.
            task_i: Index of the current task.
            metric: Uncertainty metric ('entropy', 'margin', 'least_confidence').
            ood_indices: Indices of OOD samples (used if args.ood_method is specified).
            ood_scores: OOD scores for the samples (used if args.ood_method is specified).
        """
        if self.args.uncertainty_type is not None:
            metric = self.args.uncertainty_type

        task = task_stream.tasks[task_i]
        (x_train, y_train) = task[0]  # y_train is not used for unlabeled data

        n_samples_current_task = x_train.shape[0]
        print('Number of samples in current task:', n_samples_current_task)

        n_samples_per_al_cycle = self.get_n_samples_per_al_cycle(n_samples_current_task)
        n_clusters = n_samples_per_al_cycle  # Number of clusters for diversity sampling

        # Initialize unlabeled indices
        idx_unlabeled = np.arange(n_samples_current_task)
        task_features = []


        # OOD detekció és kiértékelés
        if self.args.ood_method and task_i > 0:
            _, ood_scores = self.perform_ood_detection_and_evaluation(
                x_train, y_train, task_i, self.args.ood_method, classes_in_each_task
            )


        for alc in range(self.al_budget):
            print(f'Run: {run}, Task: {task_i}, AL cycle: {alc + 1} / {self.al_budget}')

            if alc == 0:
                # Az első ciklusban (alc == 0) eldöntjük, hogy OOD detekcióval vagy anélkül választunk mintákat
                if self.args.ood_method and task_i > 0: 
                    
                    
                     # AL+OOD logika
                    selected_idxs = self.ood_filter_top_ood(
                        x_train,
                        y_train,
                        idx_unlabeled,
                        task_stream,
                        n_samples_per_al_cycle,
                        run,
                        task_i,
                        ood_scores
                    )
                else:  # AL only logika
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
            # Ha van ood_method, akkor átadjuk a save_acc_to_csv-nek
            ood_method = self.args.ood_method if self.args.ood_method else ''
            self.save_acc_to_csv(accuracies, run, task_i, alc, f'_{metric}', ood_method = ood_method)

            # Feature gyűjtés mindkét esetben (AL only és AL+OOD)
            labeled_indices = np.setdiff1d(np.arange(n_samples_current_task), idx_unlabeled)
            if len(labeled_indices) > 0:
                eval_dataloader = Dataloader_from_numpy(
                    x_train[labeled_indices],
                    np.zeros(len(labeled_indices)),
                    self.batch_size,
                    shuffle=False
                )

                all_features = []
                for batch_id, (batch_x, _) in enumerate(eval_dataloader):
                    batch_x = batch_x.to(self.agent.device)
                    with torch.no_grad():
                        features = self.agent.model.feature(batch_x)
                    all_features.append(features.cpu().numpy())
                all_features = np.vstack(all_features)
                task_features.append(all_features)

        # Task features mentése
        if task_features:
            task_features_combined = np.vstack(task_features)
            self.id_features_list.append(task_features_combined)
            print(f"Task {task_i} - Stored {len(task_features_combined)} features in id_features_list.")

        return idx_unlabeled