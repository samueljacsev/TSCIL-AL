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

    # def select_with_ood_active_learn(self, x_train, idx_pool, y_train, n_samples_per_al_cycle, run, task_i, ood_method, al_metric, classes_in_each_task=None):
    #     """
    #     Universal OOD+Active selection: supports GMM, KM2, DRE, etc. for OOD ratio estimation.
    #     mpe_method: "gmm" | "km2" | "dre" (hardcoded for now)
    #     """
    #     if len(idx_pool) == 0:
    #         return np.array([], dtype=int)
        
        

    #     pool_x = x_train[idx_pool]
    #     pool_y = y_train[idx_pool]

    #     ood_indices_pool = self.perform_ood_detection_and_evaluation(
    #         pool_x,
    #         pool_y,
    #         task_i,
    #         ood_method=ood_method,
    #         classes_in_each_task=classes_in_each_task,
    #         run=run,
    #     )

    #     # Convert local pool indices to global
    #     selected_ood = idx_pool[ood_indices_pool]

    #     # Compute budgets
    #     n_ood = len(selected_ood)
    #     n_al  = n_samples_per_al_cycle - n_ood
    #     n_al = max(0, n_al)

    #     print(f"Selected {n_ood} OOD samples, {n_al} AL samples for this cycle.")

    #     # --- Active learning on remaining pool ---
    #     selected_al = []
    #     if n_al > 0:
    #         remaining = np.setdiff1d(idx_pool, selected_ood, assume_unique=False)
    #         if len(remaining) > 0:
    #             all_features, all_outputs = self.extract_features_and_outputs(x_train[remaining])
    #             k = min(n_al, len(remaining))
    #             if k > 0:
    #                 kmeans = KMeans(n_clusters=k, random_state=self.args.seed + run)
    #                 cluster_labels = kmeans.fit_predict(all_features)
    #                 uncertainties = self.compute_uncertainty(all_outputs, metric=al_metric)

    #                 for c in range(k):
    #                     c_idx = np.where(cluster_labels == c)[0]
    #                     if len(c_idx) == 0:
    #                         continue
    #                     c_unc = uncertainties[c_idx]
    #                     pick_local = c_idx[np.argmax(c_unc)]
    #                     selected_al.append(remaining[pick_local])

    #     selected_ood = np.array(selected_ood, dtype=int)
    #     selected_al = np.array(selected_al, dtype=int) if len(selected_al) else np.array([], dtype=int)
    #     selected = np.concatenate([selected_ood, selected_al])

    #     # Ensure exact budget and uniqueness
    #     if len(selected) > n_samples_per_al_cycle:
    #         selected = selected[:n_samples_per_al_cycle]
    #     if len(selected) < n_samples_per_al_cycle:
    #         remaining = np.setdiff1d(idx_pool, selected)
    #         if len(remaining) > 0:
    #             extra_needed = n_samples_per_al_cycle - len(selected)
    #             extra_scores = self.ood_detection(x_train[remaining], task_i, method=ood_method, return_scores=True)
    #             add = remaining[np.argsort(-extra_scores)[:extra_needed]]
    #             selected = np.concatenate([selected, add])

    #     return selected.astype(int)


    def select_with_ood_active_learn(self, x_train, idx_pool, y_train,
                                    n_samples_per_al_cycle, run, task_i,
                                    ood_method, al_metric,
                                    classes_in_each_task=None):
        """
        Combined OOD + Active Learning selection.
        - Uses estimated OOD ratio (from threshold-based evaluation) to determine
        how many OOD vs. AL samples to select.
        - Selects top OOD scores according to that ratio.
        """
        if len(idx_pool) == 0:
            print("[WARN] Empty pool passed to select_with_ood_active_learn().")
            return np.array([], dtype=int)

        # --- 1. Run OOD detection and evaluation ---
        pool_x = x_train[idx_pool]
        pool_y = y_train[idx_pool]

        ood_indices_pool, ood_scores_pool, est_ratio = self.perform_ood_detection_and_evaluation(
            pool_x,
            pool_y,
            task_i,
            ood_method=ood_method,
            classes_in_each_task=classes_in_each_task,
            run=run,
        )

        # --- 2. Compute desired budget split ---
        est_ratio = float(np.clip(est_ratio, 0.0, 1.0))
        #n_ood_target = int(round(est_ratio * n_samples_per_al_cycle))
        #n_ood_target = max(0, min(n_ood_target, n_samples_per_al_cycle))
        n_ood_target = int(round(0.2 * n_samples_per_al_cycle))
        n_al_target = n_samples_per_al_cycle - n_ood_target

        print(f"\n[INFO] Task {task_i} - Active Learning cycle budget:")
        print(f"  • Total budget = {n_samples_per_al_cycle}")
        print(f"  • Estimated OOD ratio = {est_ratio:.4f}")
        print(f"  • Target OOD samples = {n_ood_target}")
        print(f"  • Target AL samples  = {n_al_target}")

        # --- 3. Randomly sample from detected OOD indices (for diversity, not extremes) ---
        if n_ood_target > 0 and len(ood_indices_pool) > 0:
            # Use detected OOD indices, not top scores (avoids selecting only extreme outliers)
            n_ood_available = min(n_ood_target, len(ood_indices_pool))
            np.random.seed(self.args.seed + run + task_i)  # Reproducible random sampling
            selected_ood_local = np.random.choice(ood_indices_pool, size=n_ood_available, replace=False)
            selected_ood = idx_pool[selected_ood_local]
        else:
            selected_ood = np.array([], dtype=int)

        print(f"[DEBUG] Detected {len(ood_indices_pool)} OOD above threshold, "
            f"randomly selected {len(selected_ood)} for diversity (not top scores).")
        if len(selected_ood) > 0:
            print(f"[DEBUG] OOD selected class IDs: {np.unique(y_train[selected_ood])}")

        # --- 4. Active Learning selection for remaining pool ---
        selected_al = []
        remaining = np.setdiff1d(idx_pool, selected_ood, assume_unique=False)

        if n_al_target > 0 and len(remaining) > 0:
            print(f"[DEBUG] Running AL selection on remaining {len(remaining)} samples...")
            all_features, all_outputs = self.extract_features_and_outputs(x_train[remaining])
            k = min(n_al_target, len(remaining))

            if k > 0:
                kmeans = KMeans(n_clusters=k, random_state=self.args.seed + run)
                cluster_labels = kmeans.fit_predict(all_features)
                uncertainties = self.compute_uncertainty(all_outputs, metric=al_metric)

                for c in range(k):
                    c_idx = np.where(cluster_labels == c)[0]
                    if len(c_idx) == 0:
                        continue
                    c_unc = uncertainties[c_idx]
                    pick_local = c_idx[np.argmax(c_unc)]
                    selected_al.append(remaining[pick_local])
        else:
            print(f"[INFO] Skipping AL selection (remaining={len(remaining)}, n_al_target={n_al_target}).")

        # --- 5. Combine selections ---
        selected_ood = np.array(selected_ood, dtype=int)
        selected_al = np.array(selected_al, dtype=int) if len(selected_al) else np.array([], dtype=int)
        selected = np.concatenate([selected_ood, selected_al])

        # --- 6. Ensure correct total count and uniqueness ---
        selected = np.unique(selected)
        if len(selected) > n_samples_per_al_cycle:
            selected = selected[:n_samples_per_al_cycle]
            print(f"[WARN] Truncated selection to match budget ({n_samples_per_al_cycle}).")

        if len(selected) < n_samples_per_al_cycle:
            remaining = np.setdiff1d(idx_pool, selected)
            if len(remaining) > 0:
                extra_needed = n_samples_per_al_cycle - len(selected)
                print(f"[INFO] Filling missing {extra_needed} samples randomly from remaining pool.")
                # Random sampling for diversity (not top scores)
                np.random.seed(self.args.seed + run + task_i + 1)
                extra_needed_actual = min(extra_needed, len(remaining))
                add = np.random.choice(remaining, size=extra_needed_actual, replace=False)
                selected = np.concatenate([selected, add])

        # --- 7. Final debug info ---
        print(f"[FINAL] Task {task_i}: total selected = {len(selected)} "
            f"({len(selected_ood)} OOD + {len(selected_al)} AL)")
        print(f"[FINAL] Unique selected classes: {np.unique(y_train[selected])}\n")

        return selected.astype(int)




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
        task_labels = []  # Collect labels per cycle, combine at task end
            

        for alc in range(self.al_budget):
            print(f'Run: {run}, Task: {task_i}, AL cycle: {alc + 1} / {self.al_budget}')
            
            if alc == 0:
                if self.args.ood_method and task_i > 0:
                    # Stage-1 simple path: GMM-based ratio and threshold on OOD scores, no memory, no ground-truth split
                    selected_idxs = self.select_with_ood_active_learn(
                        x_train=x_train,
                        y_train=y_train,
                        idx_pool=idx_unlabeled,
                        n_samples_per_al_cycle=n_samples_per_al_cycle,
                        run=run,
                        task_i=task_i,
                        ood_method=self.args.ood_method,
                        al_metric=self.args.uncertainty_type or metric,
                        classes_in_each_task=classes_in_each_task
                    )

                    # Optional: sanity check labels
                    selected_labels = y_train[selected_idxs]
                    print(f"############################################ Task {task_i} - Classes in selected samples: {np.unique(selected_labels)}")

                else:  # HERE WE SHOULD USE ACTIVE LEARNING WITH DIVERSITY
                    # np.random.seed(self.args.seed + run)
                    # np.random.shuffle(idx_unlabeled)
                    # selected_idxs = idx_unlabeled[:n_samples_per_al_cycle]
                                # ---- Standard Uncertainty + Diversity selection ----
                    all_features, all_outputs = self.extract_features_and_outputs(x_train[idx_unlabeled])

                    # Cluster the feature embeddings for diversity
                    kmeans = KMeans(n_clusters=n_clusters, random_state=self.args.seed + run)
                    cluster_labels = kmeans.fit_predict(all_features)

                    # Compute uncertainty
                    uncertainties = self.compute_uncertainty(all_outputs, metric=metric)

                    # Select most uncertain sample from each cluster
                    selected_idxs = []
                    for cluster in range(n_clusters):
                        cluster_indices = np.where(cluster_labels == cluster)[0]
                        if len(cluster_indices) == 0:
                            continue
                        cluster_uncertainties = uncertainties[cluster_indices]
                        slct_idx = cluster_indices[np.argmax(cluster_uncertainties)]
                        selected_idxs.append(idx_unlabeled[slct_idx])

                    selected_idxs = np.array(selected_idxs[:n_samples_per_al_cycle])
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

            accuracies = self.agent.evaluate(task_stream, alc, self.al_budget)
            ood_method = self.args.ood_method if self.args.ood_method else ''
            self.save_acc_to_csv(accuracies, run, task_i, alc, f'_{metric}', ood_method = ood_method)
            
            # store only newly labeled indices in this AL cycle
            newly_labeled = selected_idxs  
            if len(newly_labeled) > 0:
                all_features, _ = self.extract_features_and_outputs(x_train[newly_labeled])
                task_features.append(all_features)
                task_labels.append(y_train[newly_labeled])  # Collect labels per cycle

            # if alc == self.al_budget - 1:
            #      return accuracies

        # Task-level storage: features and labels of all labeled samples (once per task)
        if task_features:
            task_features_combined = np.vstack(task_features)
            self.id_features_list.append(task_features_combined)
            print(f"[DEBUG] Task {task_i}: Stored {len(task_features_combined)} features in id_features_list.")
        
        if task_labels:
            task_labels_combined = np.concatenate(task_labels)
            if not hasattr(self, "id_labels_list"):
                self.id_labels_list = []
            self.id_labels_list.append(task_labels_combined)
            print(f"[DEBUG] Task {task_i}: Stored {len(task_labels_combined)} labels in id_labels_list.")

        # Store logits once per task to support ID-threshold computation across tasks
        labeled_indices_final = np.setdiff1d(np.arange(n_samples_current_task), idx_unlabeled)
        if len(labeled_indices_final) > 0:
            _, all_outputs_end = self.extract_features_and_outputs(x_train[labeled_indices_final])
            task_outputs = all_outputs_end.cpu().numpy() if hasattr(all_outputs_end, "cpu") else all_outputs_end
            if task_outputs.ndim > 2:
                task_outputs = task_outputs.reshape(task_outputs.shape[0], -1)
            if not hasattr(self, "id_outputs_list"):
                self.id_outputs_list = []
            self.id_outputs_list.append(task_outputs)
            print(f"[DEBUG] Task {task_i}: Stored {len(task_outputs)} outputs in id_outputs_list.")

        return accuracies
