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
        self.id_ood_scores_list = []
        
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
        

    @abstractmethod
    def active_learn_task(self, run, task_stream, task_i, classes_in_each_task=None):
        """
        active_learn_task: Abstract method to be implemented by subclasses.
        
        Defines how to select the next batch of samples to be labelled.
        
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    

    def ood_filter_top_ood(self, x_train, y_train, idx_unlabeled, task_stream, n_samples_per_al_cycle, run, task_i, ood_scores):
        """
        Select samples using OOD scores with clustering for the first AL cycle.

        Args:
            x_train: Training data.
            y_train: Training labels (for validation).
            idx_unlabeled: Indices of unlabeled samples.
            task_stream: Task stream object.
            n_samples_per_al_cycle: Number of samples to select per AL cycle.
            run: Random seed for reproducibility.
            task_i: Task index.
            ood_scores: OOD scores for clustering.

        Returns:
            selected_idxs: Indices of selected samples.
        """
        # Step 1: Extract embeddings for all samples in x_train
        eval_dataloader = Dataloader_from_numpy(
            x_train,
            np.zeros(len(x_train)),  # Dummy labels
            self.batch_size,
            shuffle=False
        )

        all_features = []
        for batch_id, (batch_x, _) in enumerate(eval_dataloader):
            batch_x = batch_x.to(self.agent.device)
            with torch.no_grad():
                features = self.agent.model.feature(batch_x)  # Feature extraction
                features = features.cpu().numpy()
            all_features.append(features)
        all_features = np.vstack(all_features)

        # Step 2: Cluster the embeddings
        n_clusters = n_samples_per_al_cycle
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.args.seed + run)
        cluster_labels = kmeans.fit_predict(all_features)

        # Step 3: Calculate average OOD score for each cluster
        cluster_ood_scores_avg = np.zeros(n_clusters)
        for cluster in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            if len(cluster_indices) > 0:
                cluster_ood_scores = ood_scores[cluster_indices]
                cluster_ood_scores_avg[cluster] = np.mean(cluster_ood_scores)

        # Step 4: Determine a threshold for cluster selection
        ood_threshold = np.percentile(ood_scores, 30)

        # Step 5: Select clusters with average OOD score above the threshold
        valid_clusters = np.where(cluster_ood_scores_avg > ood_threshold)[0]
        if len(valid_clusters) == 0:
            valid_clusters = np.argsort(cluster_ood_scores_avg)[-min(n_samples_per_al_cycle, n_clusters):]

        # Step 6: Distribute the n_samples_per_al_cycle across valid clusters
        selected_indices = []
        num_valid_clusters = len(valid_clusters)
        if num_valid_clusters > 0:
            samples_per_cluster = max(1, n_samples_per_al_cycle // num_valid_clusters)
            remaining_samples = n_samples_per_al_cycle % num_valid_clusters

            for i, cluster in enumerate(valid_clusters):
                cluster_indices = np.where(cluster_labels == cluster)[0]
                if len(cluster_indices) > 0:
                    cluster_ood_scores = ood_scores[cluster_indices]
                    sorted_indices = np.argsort(-cluster_ood_scores)
                    num_samples = samples_per_cluster + (1 if i < remaining_samples else 0)
                    num_samples = min(num_samples, len(cluster_indices))
                    selected_cluster_indices = cluster_indices[sorted_indices[:num_samples]]
                    selected_indices.extend(selected_cluster_indices)

        # Step 7: If not enough samples, fill with highest OOD scores
        if len(selected_indices) < n_samples_per_al_cycle:
            remaining_indices = np.setdiff1d(np.arange(len(x_train)), selected_indices)
            remaining_ood_scores = ood_scores[remaining_indices]
            additional_indices = remaining_indices[np.argsort(-remaining_ood_scores)[:n_samples_per_al_cycle - len(selected_indices)]]
            selected_indices.extend(additional_indices)

        # Step 8: Convert to numpy array and ensure exact number of samples
        selected_indices = np.array(selected_indices)[:n_samples_per_al_cycle]
        selected_idxs = np.array(selected_indices)

        # Validate selected labels
        selected_labels = y_train[selected_indices]
        if np.any(selected_labels < 0) or np.any(selected_labels >= task_stream.n_classes):
            print(f"Warning: Invalid labels found in selected indices: {np.unique(selected_labels)}")
            valid_mask = (selected_labels >= 0) & (selected_labels < task_stream.n_classes)
            selected_indices = selected_indices[valid_mask]
            selected_idxs = selected_indices[:n_samples_per_al_cycle]

        print(f"############################################Task {task_i} - Selected {len(selected_idxs)} samples using OOD scores with clustering.")
        print(f"############################################Task {task_i} - Classes in selected samples: {np.unique(y_train[selected_idxs])}")

        return selected_idxs
    

    def ood_detection(self, x_data, task_i, method='entropy', return_scores=False):
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
                # MSP (Maximum Softmax Probability) alapú OOD detekció
                softmax_scores = torch.softmax(outputs, dim=1)
                ood_scores = -torch.max(softmax_scores, dim=1)[0]  # Negatív, mert alacsonyabb MSP -> OOD
                ood_scores = ood_scores.cpu().numpy()

            elif method == 'energy':
                # Energy Score alapú OOD detekció
                ood_scores = -torch.logsumexp(outputs, dim=1)  # Negatív, mert alacsonyabb energia -> OOD
                ood_scores = ood_scores.cpu().numpy()

            elif method == 'mahalanobis':
                # Mahalanobis-távolság alapú OOD detekció
                if task_i == 0 or len(self.id_features_list) == 0:
                    raise ValueError("Mahalanobis distance requires at least one previous task with stored features.")

                # Debug: Ellenőrizzük az id_features_list elemeit
                for i, feat in enumerate(self.id_features_list[:task_i]):
                    if not isinstance(feat, np.ndarray):
                        print(f"Warning: id_features_list[{i}] is not a NumPy array, type: {type(feat)}")
                        if isinstance(feat, torch.Tensor):
                            feat = feat.cpu().numpy()
                            self.id_features_list[i] = feat

                # Összes ID feature összevonása az előző taskokból
                id_features = np.vstack(self.id_features_list[:task_i])

                if len(id_features) == 0:
                    raise ValueError("No in-distribution features found for Mahalanobis distance calculation.")

                # ID minták eloszlásának becslése
                mu = np.mean(id_features, axis=0)  # Átlag (N, D) -> (D,)
                cov = np.cov(id_features, rowvar=False)  # Kovariancia mátrix (D, D)

                # Numerikus stabilitás érdekében kis zajt adunk a kovariancia mátrixhoz
                cov += np.eye(cov.shape[0]) * 1e-6

                # Kovariancia mátrix inverzének kiszámítása
                cov_inv = np.linalg.inv(cov)

                # Mahalanobis-távolság kiszámítása minden mintára
                ood_scores = np.zeros(len(x_data))
                for i in range(len(x_data)):
                    diff = features[i] - mu  # (D,)
                    ood_scores[i] = np.sqrt(diff.T @ cov_inv @ diff)  # (D,) @ (D, D) @ (D,) = skaláris érték


            else:
                raise ValueError(f"Unknown OOD detection method: {method}")

            if return_scores:
                return ood_scores

            #threshold = np.median(ood_scores)
            percentile = 70
            threshold = np.percentile(ood_scores, percentile)
            ood_mask = (ood_scores > threshold).astype(int)
            ood_indices = np.where(ood_mask)[0]

            return ood_indices
    
    def perform_ood_detection_and_evaluation(self, x_train_current, y_train_current, task_i, ood_method, classes_in_each_task, run):
        """
        Performs OOD detection, computes ROC AUC score, and prints confusion matrix.

        Args:
            x_train_current: Current training data.
            y_train_current: Current training labels.
            task_i: Index of the current task.
            ood_method: Method for OOD detection (e.g., 'energy').
            classes_in_each_task: List of unique classes for each task.

        Returns:
            ood_indices: Indices of detected OOD samples.
            ood_scores: OOD scores for all samples.
        """
        # OOD detekció az aktuális task adathalmazán
        ood_scores = self.ood_detection(x_train_current, task_i, method=ood_method, return_scores=True)

        # Valódi címkék létrehozása: az új osztályok OOD-ként vannak jelölve
        previous_classes = []
        for i in range(task_i):
            print(f"Task {i} - Classes in each task: {classes_in_each_task[1]}")
            previous_classes.extend(classes_in_each_task[i])
        
        previous_classes = np.unique(previous_classes)
        previous_classes = previous_classes[previous_classes != -1]  # Az -1-es osztály eltávolítása

        print(f"Task {task_i} - Previous classes: {previous_classes}")

        true_labels = np.ones(len(x_train_current), dtype=int)  # Alapértelmezetten minden minta OOD
        for cls in previous_classes:
            true_labels[y_train_current == cls] = 0  # Az előző task osztályai nem OOD-k

        # print ood id ratio
        ood_id_ratio = np.sum(true_labels == 0) / len(true_labels)
        print(f"Task {task_i} - OOD/ID ratio: {ood_id_ratio:.4f}")


        # print ood labels and id labels
        print(f"Task {task_i} - OOD labels: {np.unique(y_train_current[true_labels==1])}")
        print(f"Task {task_i} - ID labels: {np.unique(y_train_current[true_labels==0])}")

        # ROC AUC score kiszámítása
        roc_auc = roc_auc_score(true_labels, ood_scores)
        print(f"Task {task_i} - ROC AUC score for OOD detection: {roc_auc:.4f}")

        # OOD detekció az aktuális task adathalmazán
        ood_indices = self.ood_detection(x_train_current, task_i, method=ood_method)

        # Predikált címkék meghatározása az ood_indices alapján
        pred_labels = np.zeros(len(x_train_current), dtype=int)  # Alapértelmezetten minden minta ID
        pred_labels[ood_indices] = 1  # Az ood_indices-ben lévő minták OOD-k

        # Confusion matrix kiszámítása
        true_positive = np.sum((true_labels == 1) & (pred_labels == 1))  # Valódi OOD, predikált OOD
        false_negative = np.sum((true_labels == 1) & (pred_labels == 0))  # Valódi OOD, predikált ID
        false_positive = np.sum((true_labels == 0) & (pred_labels == 1))  # Valódi ID, predikált OOD
        true_negative = np.sum((true_labels == 0) & (pred_labels == 0))  # Valódi ID, predikált ID

        # Confusion matrix kiíratása
        print(f"Task {task_i} - Confusion Matrix for OOD Detection:")
        print(f"True OOD (1) predicted as OOD (1): {true_positive}")
        print(f"True OOD (1) predicted as ID (0): {false_negative}")
        print(f"True ID (0) predicted as OOD (1): {false_positive}")
        print(f"True ID (0) predicted as ID (0): {true_negative}")

        print(f"Task {task_i} - Classes in OOD samples: {np.unique(y_train_current[ood_indices])}")
        print(f"Task {task_i} - Number of OOD samples detected: {len(ood_indices)} \n \n")
        
        # Mentés a CSV fájlba
        self.save_ood_metrics(x_train_current, y_train_current, task_i, ood_method, ood_scores, ood_indices, classes_in_each_task, run)
        return ood_indices, ood_scores
    


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


