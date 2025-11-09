# -*- coding: UTF-8 -*-
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import abc
import numpy as np
import os
from abc import abstractmethod
from types import SimpleNamespace
from agents.base import BaseLearner
from utils.data import Dataloader_from_numpy
from utils.metrics import compute_performance
from result.utils import save_acc_to_csv
import pandas as pd
from sklearn.mixture import GaussianMixture



class BaseSampler(nn.Module, metaclass=abc.ABCMeta):
    """
    Base class for Samplers.
    """

    def __init__(self, 
                 agent: BaseLearner, 
                 exp_args:SimpleNamespace, 
                 args:SimpleNamespace, 
                 name: str='base'):
        super().__init__()
        
        self.name = name
        self.args = args
        self.agent = agent
        self.batch_size = agent.batch_size
        self.al_budget = exp_args.al_budget
        self.al_total = exp_args.al_total
        self.id_features_list = []
        self.id_outputs_list = []
        self.id_ood_scores_list = []
        self.id_labels_list = []

        print('AL strategy:', self.name)
        print('Batch size:', self.batch_size)
        

    def get_n_samples_per_al_cycle(self, n_samples_current_task):
        """
        get_n_of_al_cycles: Calculate the number of active learning cycles.
        
        Returns:
            Number of active learning cycles.
        """

        # Number of samples to label per AL cycle 
        n_samples_per_al_cycle = n_samples_current_task // self.al_total
        print('# Number of samples in the AL cycle:', n_samples_per_al_cycle)
        return n_samples_per_al_cycle #+ 1 # buffer was truncated
    
    def save_acc_to_csv(self, accs_data, run, task, cycle, ext='', ood_method=''):
        mean_excd_0 = np.mean(accs_data[accs_data != 0], axis=0)
        print(f'Acc_vector: {accs_data}, Mean: {mean_excd_0} ')
        ood_suffix = f'_{ood_method}' if ood_method else ''  # Ha van OOD metódus, illesszük be a nevét
        fn = f'{self.name}{ood_suffix}{ext}_cycle_{self.al_budget}_per_{self.al_total}_{self.args.data}.csv'
        fn = os.path.join('result',self.args.data , fn)
        save_acc_to_csv(accs_data, run, task, cycle, filename=fn)
    
    
    def calculate_metrcs(self, acc_matrix):
        acc_matrix = [np.array(acc_matrix)]
        acc_matrix = np.array(acc_matrix)
        
        avg_end_acc, avg_end_fgt, avg_cur_acc, avg_acc = compute_performance(acc_matrix)
        print('Average end accuracy:', avg_end_acc)
        print('Average end forgetting:', avg_end_fgt)
        print('Average current accuracy:', avg_cur_acc)
        print('Average accuracy:', avg_acc)
        #print('Average BWT+:', avg_bwtp)

    def extract_features_and_outputs(self, x_array):

        # Step 1: Extract embeddings for all unlabeled samples
        eval_dataloader = Dataloader_from_numpy(
        x_array,
        np.zeros(len(x_array)),  # Dummy labels
        self.batch_size,
        shuffle=False)

        all_features, all_outputs = [], []
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

        return all_features, all_outputs
        

    @abstractmethod
    def active_learn_task(self, run, task_stream, task_i, classes_in_each_task=None):
        """
        active_learn_task: Abstract method to be implemented by subclasses.
        
        Defines how to select the next batch of samples to be labelled.
        
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    

    def save_ood_debug_data(self, task_i, run, method, stage, scores, labels=None, threshold=None, save_dir="ood_debug_logs"):
        """
        Save OOD score distributions, thresholds, and metadata for later visualization.
        stage: 'id' (from previous tasks) or 'current' (mixture from current task)
        """
        os.makedirs(save_dir, exist_ok=True)

        # Get dataset name and shuffle ratio from args
        dataset_name = getattr(self.args, 'data', 'unknown')
        shuffle_ratio = getattr(self.args, 'shuffle_ratio', 0.0)

        df = pd.DataFrame({
            "dataset": dataset_name,
            "shuffle_ratio": shuffle_ratio,
            "task": task_i,
            "run": run,
            "method": method,
            "stage": stage,
            "score": scores,
        })

        if labels is not None:
            df["true_label"] = labels  # 0=ID, 1=OOD if available

        if threshold is not None:
            df["threshold"] = threshold

        # Include dataset and shuffle_ratio in filename for better traceability
        csv_path = os.path.join(save_dir, f"{dataset_name}_{shuffle_ratio}_run{run}_task{task_i}_{method}_{stage}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[LOG] Saved OOD debug data: {csv_path} ({len(scores)} samples)")

    
    def estimate_ood_threshold(self, task_i, method='energy', percentile=99):
        """
        Estimate an OOD threshold from previously labeled ID data only.
        - For MSP/Energy: compute 1D ID scores per task from stored logits and concatenate.
        - For Mahalanobis: compute distances from stacked ID features.
        Returns: scalar threshold (float) or None if no prior ID data.
        """
        if task_i == 0:
            return None

        if method == 'mahalanobis':
            # --- Collect previous ID features and labels ---
            if len(self.id_features_list) == 0 or len(self.id_labels_list) == 0:
                return None
            try:
                id_features = np.vstack(self.id_features_list[:task_i])
                id_labels = np.concatenate(self.id_labels_list[:task_i])
            except Exception as e:
                print(f"[WARN] Could not stack ID features/labels: {e}")
                return None

            print(f"[Mahalanobis] ID features shape: {id_features.shape}, labels shape: {id_labels.shape}")
            
            # --- Compute per-class means and shared covariance ---
            classes = np.unique(id_labels)
            print(f"[Mahalanobis] Computing stats for {len(classes)} classes: {classes}")
            class_means = {c: np.mean(id_features[id_labels == c], axis=0) for c in classes}
            cov = np.cov(id_features, rowvar=False) + np.eye(id_features.shape[1]) * 1e-6
            cov_inv = np.linalg.inv(cov)

            # --- Compute per-sample Mahalanobis (min distance to any class mean) ---
            id_scores = []
            for feat in id_features:
                dists = [np.sqrt((feat - class_means[c]).T @ cov_inv @ (feat - class_means[c])) for c in classes]
                id_scores.append(np.min(dists))
            id_scores = np.array(id_scores)
            
            print(f"[Mahalanobis] ID score stats: min={np.min(id_scores):.4f}, max={np.max(id_scores):.4f}, "
                  f"mean={np.mean(id_scores):.4f}, median={np.median(id_scores):.4f}, std={np.std(id_scores):.4f}")
            
        if method == 'cosine':
            # --- Collect previous ID features + labels ---
            if len(self.id_features_list) == 0 or len(self.id_labels_list) == 0:
                return None
            try:
                id_features = np.vstack(self.id_features_list[:task_i])
                id_labels = np.concatenate(self.id_labels_list[:task_i])
            except Exception as e:
                print(f"[WARN] Could not stack ID features/labels: {e}")
                return None

            # --- Normalize features ---
            id_features = id_features / np.linalg.norm(id_features, axis=1, keepdims=True)

            # --- Per-class mean directions ---
            classes = np.unique(id_labels)
            class_means = {c: np.mean(id_features[id_labels == c], axis=0) for c in classes}
            class_means = {c: m / np.linalg.norm(m) for c, m in class_means.items()}

            # --- Compute cosine distances for ID samples ---
            id_scores = []
            for f, label in zip(id_features, id_labels):
                sim = np.dot(f, class_means[label])
                id_scores.append(1 - sim)  # cosine distance
            id_scores = np.array(id_scores)
            print(f"[Cosine] ID score stats: min={np.min(id_scores):.4f}, max={np.max(id_scores):.4f}, "
                    f"mean={np.mean(id_scores):.4f}, median={np.median(id_scores):.4f}, std={np.std(id_scores):.4f}")
            
        elif method == 'knn':
            # === kNN-based OOD threshold estimation ===
            from sklearn.neighbors import NearestNeighbors

            if len(self.id_features_list) == 0:
                return None
            id_features = np.vstack(self.id_features_list[:task_i])


            # Normalize for stability
            id_features = id_features / np.linalg.norm(id_features, axis=1, keepdims=True)

            # Fit kNN on ID features (default k=5, adjusted if fewer samples)
            k = 3  # Default number of neighbors
            k = min(k, len(id_features) - 1)
            knn = NearestNeighbors(n_neighbors=k + 1)  # +1 to exclude self
            knn.fit(id_features)

            # Compute mean distance to k nearest neighbors (excluding self)
            distances, _ = knn.kneighbors(id_features)
            mean_dists = np.mean(distances[:, 1:], axis=1)
            id_scores = mean_dists

            print(f"[kNN ID stats] k={k}, min={np.min(id_scores):.4f}, max={np.max(id_scores):.4f}, mean={np.mean(id_scores):.4f}")

        else:
            # MSP / Energy from stored logits per task (avoid stacking logits across tasks)
            if not hasattr(self, 'id_outputs_list') or len(self.id_outputs_list) == 0:
                return None
            id_score_chunks = []
            for logits_np in self.id_outputs_list[:task_i]:
                if logits_np is None or len(logits_np) == 0:
                    continue
                logits = torch.tensor(logits_np, dtype=torch.float32)
                if method == 'msp':
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    scores = -np.sum(probs * np.log(probs + 1e-10), axis=1)
                    id_score_chunks.append(scores)
                elif method == 'energy':
                    scores = -torch.logsumexp(logits, dim=1)
                    id_score_chunks.append(scores.cpu().numpy())
                else:
                    raise ValueError(f"Unsupported method: {method}")
            if len(id_score_chunks) == 0:
                return None
            id_scores = np.concatenate(id_score_chunks, axis=0)

        threshold = float(np.percentile(id_scores, percentile))
        print(f"[OOD] ID-based threshold ({percentile}th): {threshold:.6f}")

        # --- DEBUG SAVE for ID scores ---
        try:
            self.save_ood_debug_data(
                task_i=task_i,
                run=getattr(self.args, "run", 0),
                method=method,
                stage="id",
                scores=id_scores,
                threshold=threshold
            )
        except Exception as e:
            print(f"[WARN] Could not save ID OOD debug data: {e}")

        return threshold

    

    # def ood_filter_top_ood(self, x_train, n_samples_per_al_cycle, run, ood_scores):
    #     """
    #     Select samples using OOD scores with clustering for the first AL cycle.

    #     Args:
    #         x_train: Training data.
    #         n_samples_per_al_cycle: Number of samples to select per AL cycle.
    #         run: Random seed for reproducibility.
    #         ood_scores: OOD scores for clustering.

    #     Returns:
    #         selected_idxs: Indices of selected samples.
    #     """

    #     all_features, _ = self.extract_features_and_outputs(x_train)

    #     # Step 2: Cluster the embeddings
    #     n_clusters = n_samples_per_al_cycle
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=self.args.seed + run)
    #     cluster_labels = kmeans.fit_predict(all_features)

    #     # Step 3: Calculate average OOD score for each cluster
    #     cluster_ood_scores_avg = np.zeros(n_clusters)
    #     for cluster in range(n_clusters):
    #         cluster_indices = np.nonzero(cluster_labels == cluster)[0]
    #         if len(cluster_indices) > 0:
    #             cluster_ood_scores = ood_scores[cluster_indices]
    #             cluster_ood_scores_avg[cluster] = np.mean(cluster_ood_scores)

    #     # Step 4: Determine a threshold for cluster selection
    #     ood_threshold = np.percentile(ood_scores, 30)

    #     # Step 5: Select clusters with average OOD score above the threshold
    #     valid_clusters = np.nonzero(cluster_ood_scores_avg > ood_threshold)[0]
    #     if len(valid_clusters) == 0:
    #         valid_clusters = np.argsort(cluster_ood_scores_avg)[-min(n_samples_per_al_cycle, n_clusters):]

    #     # Step 6: Distribute the n_samples_per_al_cycle across valid clusters
    #     selected_indices = []
    #     num_valid_clusters = len(valid_clusters)
    #     if num_valid_clusters > 0:
    #         samples_per_cluster = max(1, n_samples_per_al_cycle // num_valid_clusters)
    #         remaining_samples = n_samples_per_al_cycle % num_valid_clusters

    #         for i, cluster in enumerate(valid_clusters):
    #             cluster_indices = np.nonzero(cluster_labels == cluster)[0]
    #             if len(cluster_indices) > 0:
    #                 cluster_ood_scores = ood_scores[cluster_indices]
    #                 sorted_indices = np.argsort(-cluster_ood_scores)
    #                 num_samples = samples_per_cluster + (1 if i < remaining_samples else 0)
    #                 num_samples = min(num_samples, len(cluster_indices))
    #                 selected_cluster_indices = cluster_indices[sorted_indices[:num_samples]]
    #                 selected_indices.extend(selected_cluster_indices)

    #     # Step 7: If not enough samples, fill with highest OOD scores
    #     if len(selected_indices) < n_samples_per_al_cycle:
    #         remaining_indices = np.setdiff1d(np.arange(len(x_train)), selected_indices)
    #         remaining_ood_scores = ood_scores[remaining_indices]
    #         additional_indices = remaining_indices[np.argsort(-remaining_ood_scores)[:n_samples_per_al_cycle - len(selected_indices)]]
    #         selected_indices.extend(additional_indices)

    #     # Step 8: Convert to numpy array and ensure exact number of samples
    #     selected_indices = np.array(selected_indices)[:n_samples_per_al_cycle]
    #     selected_idxs = np.array(selected_indices)

    #     return selected_idxs
    

    def ood_detection(self, x_data, task_i, method='entropy', run=0, return_scores=False):
        """
        OOD detekció különböző módszerekkel.
        method: 'entropy', 'msp', 'energy', 'odin'
        """
        self.agent.model.eval()
        dataloader = Dataloader_from_numpy(
            x_data, 
            np.zeros(len(x_data)),  # Dummy labels
            self.batch_size,
            shuffle=False
        )

        all_outputs = []
        all_features = []
        for batch_id, (batch_x, _) in enumerate(dataloader):
            batch_x = batch_x.to(self.agent.device)
            with torch.no_grad():
                outputs = self.agent.model(batch_x)  # Logits: (N, C)
                features = self.agent.model.feature(batch_x)  # Features: (N, D)
            all_outputs.append(outputs)
            all_features.append(features)
        outputs = torch.cat(all_outputs, dim=0)
        features = torch.cat(all_features, dim=0).cpu().numpy()

        # Különböző OOD detekciós módszerek
        if method == 'entropy':
            # Entropia alapú OOD detekció (eredeti módszer)
            softmax_scores = torch.softmax(outputs, dim=1)
            ood_scores = -torch.sum(softmax_scores * torch.log(softmax_scores + 1e-6), dim=1)
            ood_scores = ood_scores.cpu().numpy()

        elif method == 'msp':
            # MSP: define OOD score as (1 - max softmax) so higher => more OOD
            softmax_scores = torch.softmax(outputs, dim=1).cpu().numpy()
            ood_scores = -np.sum(softmax_scores * np.log(softmax_scores + 1e-10), axis=1)
            # ood_scores = 1.0 - torch.max(softmax_scores, dim=1)[0]
            # ood_scores = ood_scores.cpu().numpy()

        elif method == 'energy':
            # Energy Score alapú OOD detekció
            ood_scores = -torch.logsumexp(outputs, dim=1)  # Negatív, mert alacsonyabb energia -> OOD
            ood_scores = ood_scores.cpu().numpy()

        elif method == 'mahalanobis':
            if task_i == 0 or len(self.id_features_list) == 0:
                raise ValueError("Mahalanobis distance requires at least one previous task with stored features and labels.")

            # Stack previous features and labels
            id_features = np.vstack(self.id_features_list[:task_i])
            id_labels = np.concatenate(self.id_labels_list[:task_i])

            print(f"[Mahalanobis OOD Detection] ID features: {id_features.shape}, labels: {id_labels.shape}, current pool: {features.shape}")

            # Per-class statistics
            classes = np.unique(id_labels)
            print(f"[Mahalanobis OOD Detection] Classes: {classes}")
            class_means = {c: np.mean(id_features[id_labels == c], axis=0) for c in classes}
            cov = np.cov(id_features, rowvar=False) + np.eye(id_features.shape[1]) * 1e-6
            cov_inv = np.linalg.inv(cov)

            # Compute Mahalanobis distance per sample to nearest class mean
            ood_scores = np.zeros(len(features))
            for i in range(len(features)):
                dists = [np.sqrt((features[i] - class_means[c]).T @ cov_inv @ (features[i] - class_means[c])) for c in classes]
                ood_scores[i] = np.min(dists)
            
            print(f"[Mahalanobis OOD Detection] Score range: [{np.min(ood_scores):.4f}, {np.max(ood_scores):.4f}], "
                  f"mean={np.mean(ood_scores):.4f}, median={np.median(ood_scores):.4f}")
            print(f"[Mahalanobis OOD Detection] 10/50/90 percentiles: {np.percentile(ood_scores, [10, 50, 90])}")
        
        elif method == 'cosine':
            if task_i == 0 or len(self.id_features_list) == 0 or len(self.id_labels_list) == 0:
                raise ValueError("Cosine OOD requires previous task features and labels.")

            id_features = np.vstack(self.id_features_list[:task_i])
            id_labels = np.concatenate(self.id_labels_list[:task_i])

            # --- Normalize both ID and current features ---
            id_features = id_features / np.linalg.norm(id_features, axis=1, keepdims=True)
            features = features / np.linalg.norm(features, axis=1, keepdims=True)

            # --- Compute per-class mean direction ---
            classes = np.unique(id_labels)
            class_means = {c: np.mean(id_features[id_labels == c], axis=0) for c in classes}
            class_means = {c: m / np.linalg.norm(m) for c, m in class_means.items()}

            # --- Compute 1 - cosine similarity to nearest class ---
            ood_scores = np.zeros(len(features))
            for i, f in enumerate(features):
                sims = [np.dot(f, class_means[c]) for c in classes]
                ood_scores[i] = 1 - np.max(sims)

            print(f"[Cosine OOD Detection] Score range: [{np.min(ood_scores):.4f}, {np.max(ood_scores):.4f}], "
                f"mean={np.mean(ood_scores):.4f}, median={np.median(ood_scores):.4f}")
        
        elif method == 'knn':
            # --- kNN-based OOD Detection ---
            from sklearn.neighbors import NearestNeighbors

            if len(self.id_features_list) == 0:
                raise ValueError("kNN OOD requires previous ID features.")

            id_features = np.vstack(self.id_features_list[:task_i])
            id_features = id_features / np.linalg.norm(id_features, axis=1, keepdims=True)
            features = features / np.linalg.norm(features, axis=1, keepdims=True)

            # Fit kNN on ID samples (default k=5, adjusted if fewer samples available)
            k = 3  # Default number of neighbors
            k = min(k, len(id_features))
            knn = NearestNeighbors(n_neighbors=k)
            knn.fit(id_features)

            # Compute mean distance to k nearest neighbors
            distances, _ = knn.kneighbors(features)
            ood_scores = np.mean(distances, axis=1)

            print(f"[kNN OOD Detection] k={k}, Score range: [{np.min(ood_scores):.4f}, {np.max(ood_scores):.4f}], "
                f"mean={np.mean(ood_scores):.4f}, median={np.median(ood_scores):.4f}")

        else:
            raise ValueError(f"Unknown OOD detection method: {method}")

        
        return ood_scores

        # # Dynamically estimate threshold via GMM on OOD scores (simple 2-comp; quantile threshold)
        # threshold, est_ratio = self.estimate_ood_threshold(ood_scores, run)
        # frac_above = float(np.mean(ood_scores >= threshold))
        # print(f"[OOD] method={method} thr={threshold:.6f} est_ratio={est_ratio:.4f} frac_above={frac_above:.4f}")
        # ood_mask = (ood_scores >= threshold).astype(int)
        # ood_indices = np.nonzero(ood_mask)[0]

        # return ood_indices

    # def perform_ood_detection_and_evaluation(self, x_train_current, y_train_current, task_i, ood_method, classes_in_each_task, run):
    #     """
    #     Performs OOD detection, computes ROC AUC score, and prints confusion matrix.

    #     Args:
    #         x_train_current: Current training data.
    #         y_train_current: Current training labels.
    #         task_i: Index of the current task.
    #         ood_method: Method for OOD detection (e.g., 'energy').
    #         classes_in_each_task: List of unique classes for each task.

    #     Returns:
    #         ood_indices: Indices of detected OOD samples.
    #         ood_scores: OOD scores for all samples.
    #     """
    #     # OOD detekció az aktuális task adathalmazán
    #     ood_scores = self.ood_detection(x_train_current, task_i, method=ood_method, run=run, return_scores=True)

    #     # Valódi címkék létrehozása: az új osztályok OOD-ként vannak jelölve
    #     previous_classes = []
    #     for i in range(task_i):
    #         print(f"Task {i} - Classes in each task: {classes_in_each_task[1]}")
    #         previous_classes.extend(classes_in_each_task[i])
        
    #     previous_classes = np.unique(previous_classes)
    #     previous_classes = previous_classes[previous_classes != -1]  # Az -1-es osztály eltávolítása

    #     print(f"Task {task_i} - Previous classes: {previous_classes}")

    #     true_labels = np.ones(len(x_train_current), dtype=int)  # Alapértelmezetten minden minta OOD
    #     for cls in previous_classes:
    #         true_labels[y_train_current == cls] = 0  # Az előző task osztályai nem OOD-k

    #     # print ood id ratio
    #     ood_id_ratio = np.sum(true_labels == 0) / len(true_labels)
    #     print(f"Task {task_i} - OOD/ID ratio: {ood_id_ratio:.4f}")


    #     # print ood labels and id labels
    #     print(f"Task {task_i} - OOD labels: {np.unique(y_train_current[true_labels==1])}")
    #     print(f"Task {task_i} - ID labels: {np.unique(y_train_current[true_labels==0])}")

    #     # ROC AUC score kiszámítása
    #     roc_auc = roc_auc_score(true_labels, ood_scores)
    #     print(f"Task {task_i} - ROC AUC score for OOD detection: {roc_auc:.4f}")

    #     # Step 2: Estimate threshold based on previous ID data
    #     threshold = self.estimate_ood_threshold(task_i, method=ood_method, percentile=95)
    #     if threshold is None:
    #         print("[WARN] No ID data found. Using 95th percentile of current scores.")
    #         threshold = np.percentile(ood_scores, 95)

    #     # Step 3: Apply threshold to get OOD mask and indices
    #     ood_mask = (ood_scores > threshold).astype(int)
    #     ood_indices = np.where(ood_mask == 1)[0]
    #     est_ratio = float(np.mean(ood_mask))  # fraction above threshold

    #     # Predikált címkék meghatározása az ood_indices alapján
    #     pred_labels = np.zeros(len(x_train_current), dtype=int)  # Alapértelmezetten minden minta ID
    #     pred_labels[ood_indices] = 1  # Az ood_indices-ben lévő minták OOD-k

    #     # Confusion matrix kiszámítása
    #     true_positive = np.sum((true_labels == 1) & (pred_labels == 1))  # Valódi OOD, predikált OOD
    #     false_negative = np.sum((true_labels == 1) & (pred_labels == 0))  # Valódi OOD, predikált ID
    #     false_positive = np.sum((true_labels == 0) & (pred_labels == 1))  # Valódi ID, predikált OOD
    #     true_negative = np.sum((true_labels == 0) & (pred_labels == 0))  # Valódi ID, predikált ID

    #     # Confusion matrix kiíratása
    #     print(f"Task {task_i} - Confusion Matrix for OOD Detection:")
    #     print(f"True OOD (1) predicted as OOD (1): {true_positive}")
    #     print(f"True OOD (1) predicted as ID (0): {false_negative}")
    #     print(f"True ID (0) predicted as OOD (1): {false_positive}")
    #     print(f"True ID (0) predicted as ID (0): {true_negative}")

    #     print(f"Task {task_i} - Classes in OOD samples: {np.unique(y_train_current[ood_indices])}")
    #     print(f"Task {task_i} - Number of OOD samples detected: {len(ood_indices)} \n \n")
        
    #     # Mentés a CSV fájlba
    #     #self.save_ood_metrics(x_train_current, y_train_current, task_i, ood_method, ood_scores, ood_indices, classes_in_each_task, run)
    #     return ood_indices, ood_scores, est_ratio
    
    def perform_ood_detection_and_evaluation(self, x_train_current, y_train_current, task_i, ood_method, classes_in_each_task, run):
        """
        Performs OOD detection, computes ROC AUC, confusion matrix,
        and returns threshold + estimated OOD ratio.
        """

        # Step 1: Get OOD scores (no thresholding here)
        ood_scores = self.ood_detection(
            x_train_current, task_i, method=ood_method, run=run, return_scores=True
        )

        # Step 2: Estimate threshold based on previous ID data
        threshold = self.estimate_ood_threshold(task_i, method=ood_method, percentile=80)
        if threshold is None:
            print("[WARN] No ID data found. Using 95th percentile of current scores.")
            threshold = np.percentile(ood_scores, 95)

        # Step 3: Apply threshold to get OOD mask and indices
        ood_mask = (ood_scores > threshold).astype(int)
        ood_indices = np.where(ood_mask == 1)[0]
        est_ratio = float(np.mean(ood_mask))  # fraction above threshold

        # Step 4: Evaluate (for monitoring, optional)
        previous_classes = np.unique(np.concatenate(classes_in_each_task[:task_i])) if task_i > 0 else []
        true_labels = np.ones(len(y_train_current), dtype=int)
        true_labels[np.isin(y_train_current, previous_classes)] = 0

        roc_auc = roc_auc_score(true_labels, ood_scores)
        print(f"[Eval] Task {task_i} | ROC AUC: {roc_auc:.4f} | est_ratio: {est_ratio:.4f}")

        print(f"[Eval] threshold={threshold:.6f}, detected OOD={len(ood_indices)} / {len(ood_scores)}")

        # Optional: print confusion matrix
        tp = np.sum((true_labels == 1) & (ood_mask == 1))
        fn = np.sum((true_labels == 1) & (ood_mask == 0))
        fp = np.sum((true_labels == 0) & (ood_mask == 1))
        tn = np.sum((true_labels == 0) & (ood_mask == 0))
        print(f"[Confusion] TP={tp}, FN={fn}, FP={fp}, TN={tn}")

        # --- DEBUG SAVE for current task mixture ---
        try:
            self.save_ood_debug_data(
                task_i=task_i,
                run=run,
                method=ood_method,
                stage="current",
                scores=ood_scores,
                labels=true_labels,  # 0=ID, 1=OOD
                threshold=threshold
            )
        except Exception as e:
            print(f"[WARN] Could not save current OOD debug data: {e}")


        return ood_indices, ood_scores, est_ratio


    def save_ood_metrics(self, x_train_current, y_train_current, task_i, ood_method, ood_scores, ood_indices, classes_in_each_task, run):
        """
        Saves ROC AUC scores and confusion matrix metrics (TP, TN, FP, FN) to a CSV file for each OOD method.

        Args:
            x_train_current: Current training data.
            y_train_current: Current training labels.
            task_i: Index of the current task.
            ood_method: Method used for OOD detection (e.g., 'energy').
            ood_scores: OOD scores for all samples.
            ood_indices: Indices of detected OOD samples.
            classes_in_each_task: List of unique classes for each task.
            run: Run identifier for distinguishing multiple runs.
        """
        # Valódi címkék létrehozása: új osztályok OOD-ként jelölve
        previous_classes = []
        for i in range(task_i):
            previous_classes.extend(classes_in_each_task[i])
        previous_classes = np.unique(previous_classes)
        previous_classes = previous_classes[previous_classes != -1]

        true_labels = np.ones(len(x_train_current), dtype=int)  # Alapértelmezett: minden minta OOD
        for cls in previous_classes:
            true_labels[y_train_current == cls] = 0  # Előző task osztályai ID-k

        # ROC AUC score kiszámítása
        roc_auc = roc_auc_score(true_labels, ood_scores)

        # Predikált címkék az ood_indices alapján
        pred_labels = np.zeros(len(x_train_current), dtype=int)  # Alapértelmezett: minden minta ID
        pred_labels[ood_indices] = 1  # OOD minták

        # Konfúziós mátrix számítás
        true_positive = np.sum((true_labels == 1) & (pred_labels == 1))
        false_negative = np.sum((true_labels == 1) & (pred_labels == 0))
        false_positive = np.sum((true_labels == 0) & (pred_labels == 1))
        true_negative = np.sum((true_labels == 0) & (pred_labels == 0))

        # Mappa létrehozása: result/<dataset_name>/ood_metrics/
        ood_folder = os.path.join("result", self.args.data, "ood_metrics")
        os.makedirs(ood_folder, exist_ok=True)

        # CSV fájl neve az OOD metódus alapján (pl. mahalanobis.csv, msp.csv, energy.csv)
        filename = os.path.join(ood_folder, f"{ood_method}.csv")

        # Adatok előkészítése
        data = {
            "run": [run],
            "task": [task_i],
            "score": [roc_auc],
            "True_Positive": [true_positive],
            "True_Negative": [true_negative],
            "False_Positive": [false_positive],
            "False_Negative": [false_negative]
        }
        df = pd.DataFrame(data)

        # CSV fájlba mentés hozzáfűzéses módban
        if not os.path.exists(filename):
            df.to_csv(filename, index=False)
        else:
            df.to_csv(filename, mode='a', header=False, index=False)
        print(f"OOD metrics saved to {filename}")


