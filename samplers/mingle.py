from sklearn.cluster import KMeans
from types import SimpleNamespace
from agents.base import BaseLearner
from samplers.base import BaseSampler
import numpy as np
import torch
import warnings

warnings.filterwarnings('ignore')


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

    
    def pseudo_separate_ood_and_id_samples(self, run, task_stream, task_i, n_samples_per_al_cycle, y_train, x_train, metric='entropy'):
        
        # --- Step 0: Separate OOD and ID classes from ground truth (PoC) ---
        # Get classes that the model has actually seen in previous tasks (after shuffle)
        id_classes_list = []
        for prev_task_idx in range(task_i):
            prev_task = task_stream.tasks[prev_task_idx]
            prev_y_train = prev_task[0][1]  # Get labels from training data
            id_classes_list.extend(np.unique(prev_y_train))
        
        id_classes = np.unique(id_classes_list)  # Remove duplicates and sort
        
        # Find OOD and ID indices based on ground truth labels
        ood_indices = np.where(~np.isin(y_train, id_classes))[0]
        id_indices = np.where(np.isin(y_train, id_classes))[0]
        
        # Get actual class labels for OOD and ID samples
        ood_class_labels = np.unique(y_train[ood_indices]) if len(ood_indices) > 0 else []
        id_class_labels = np.unique(y_train[id_indices]) if len(id_indices) > 0 else []
        
        # Print detailed information about OOD and ID separation
        print("="*60)
        print(f"[Task {task_i}, AL Cycle 0] OOD/ID Separation Results:")
        print(f"ID classes (previously seen): {id_class_labels}")
        print(f"ID sample indices: {id_indices} (count: {len(id_indices)})")
        print(f"OOD classes (new/unseen): {ood_class_labels}")
        print(f"OOD sample indices: {ood_indices} (count: {len(ood_indices)})")
        print("="*60)

        # --- Step 1: determine counts ---
        n_ood_samples = int(self.args.shuffle_ratio * n_samples_per_al_cycle)
        n_al_samples = n_samples_per_al_cycle - n_ood_samples

        print(f"Sampling strategy: {n_ood_samples} OOD samples (random) + {n_al_samples} ID samples (AL+diversity)")

        # --- Step 2: Random sampling for OOD samples ---
        np.random.seed(self.args.seed + run)
        if len(ood_indices) < n_ood_samples:
            selected_idxs_ood = ood_indices
            print(f"Warning: Only {len(ood_indices)} OOD samples available, less than requested {n_ood_samples}")
        else:
            selected_idxs_ood = np.random.choice(ood_indices, n_ood_samples, replace=False)
        
        print(f"Selected OOD samples: {selected_idxs_ood} (count: {len(selected_idxs_ood)})")

        # --- Step 3: Active learning strategy ONLY on ID samples ---
        # Use only ID indices for clustering and AL selection
        id_pool_for_al = np.setdiff1d(id_indices, selected_idxs_ood)  # Remove any ID samples already selected as OOD
        
        print(f"ID pool for active learning: {len(id_pool_for_al)} samples")

        all_features, all_outputs = self.extract_features_and_outputs(x_train[id_pool_for_al])

        # cluster only the remaining pool
        kmeans = KMeans(n_clusters=n_al_samples, random_state=self.args.seed + run)
        cluster_labels = kmeans.fit_predict(all_features)

        uncertainties = self.compute_uncertainty(all_outputs, metric=metric)

        selected_idxs_AL = []
        for cluster in range(n_al_samples):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            cluster_uncertainties = uncertainties[cluster_indices]
            slct_idx = cluster_indices[np.argmax(-cluster_uncertainties)]
            selected_idxs_AL.append(id_pool_for_al[slct_idx])

        selected_idxs_AL = np.array(selected_idxs_AL)
        
        print(f"Selected ID samples (AL+diversity): {selected_idxs_AL} (count: {len(selected_idxs_AL)})")
        print(f"ID sample classes: {np.unique(y_train[selected_idxs_AL])}")

        # --- Step 4: merge OOD + AL selections ---
        selected_idxs = np.concatenate([selected_idxs_ood, selected_idxs_AL])

        print(f"[PoC] Total selected samples: {len(selected_idxs)} ({len(selected_idxs_ood)} OOD + {len(selected_idxs_AL)} ID)")
        print("="*60)
        
        return selected_idxs


    def active_learn_task(self, run, task_stream, task_i, metric='entropy', classes_in_each_task=None):
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
            

        for alc in range(self.al_budget):
            print(f'Run: {run}, Task: {task_i}, AL cycle: {alc + 1} / {self.al_budget}')
            
            if alc == 0:
                if self.args.ood_method and task_i > 0:
                    # OOD detection and evaluation
                    if (PSEUDO_SEPARATE := True):
                        selected_idxs = self.pseudo_separate_ood_and_id_samples(run, task_stream, task_i, n_samples_per_al_cycle, y_train, x_train, self.args.uncertainty_type)
                    else:
                        ood_indices, ood_scores = self.perform_ood_detection_and_evaluation(x_train, y_train, task_i, self.args.ood_method, classes_in_each_task, run)
                        selected_idxs = self.ood_filter_top_ood(x_train, y_train, ood_indices, task_stream, n_samples_per_al_cycle, run, task_i, ood_scores)

                            # Optional: sanity check labels
                    selected_labels = y_train[selected_idxs]
                    print(f"############################################ Task {task_i} - Selected {len(selected_idxs)} OOD samples with clustering.")
                    print(f"############################################ Task {task_i} - Classes in selected samples: {np.unique(selected_labels)}")

                else:  # AL only logika
                    np.random.seed(self.args.seed + run )
                    np.random.shuffle(idx_unlabeled)
                    selected_idxs = idx_unlabeled[:n_samples_per_al_cycle]
            else:
                all_features, all_outputs = self.extract_features_and_outputs(x_train[idx_unlabeled])
                
                # Step 2: Cluster the embeddings
                kmeans = KMeans(n_clusters=n_clusters, random_state=self.args.seed + run)
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

            # Feature gyűjtés mindkét esetben (AL only és AL+OOD)
            labeled_indices = np.setdiff1d(np.arange(n_samples_current_task), idx_unlabeled)
            if len(labeled_indices) > 0:
                all_features, _ = self.extract_features_and_outputs(x_train[labeled_indices])

        # Task features mentése
        if task_features:
            task_features_combined = np.vstack(task_features)
            self.id_features_list.append(task_features_combined)
            print(f"Task {task_i} - Stored {len(task_features_combined)} features in id_features_list.")

        return idx_unlabeled