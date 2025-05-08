# -*- coding: UTF-8 -*-
import numpy as np
from utils.data import extract_samples_according_to_labels, extract_samples_according_to_labels_with_sub, extract_n_samples_randomly
from utils.utils import load_pickle
from utils.setup_elements import n_classes, n_tasks, data_path, preset_orders, n_tasks_val, n_classes_per_task, input_size_match
import torch
import random


class IncrementalTaskStream(object):
    def __init__(self, data, scenario, cls_order, split):

        self.tasks = []
        self.data = data
        self.scenario = scenario
        self.path = data_path[data]
        self.split = split

        if self.split == 'all':  # Decide which task stream to use, Exp or Val or All
            start, end = 0, n_tasks[data]
        elif self.split == 'val':
            start, end = 0, n_tasks_val[data]
        elif self.split == 'exp':
            start = 0 if n_tasks[self.data] < 5 else n_tasks_val[data]
            end = n_tasks[data]
        else:
            raise ValueError("Incorrect task stream split")

        # Setup elements for this task stream
        self.n_tasks = end - start
        self.n_class_per_task = n_classes_per_task[data]
        self.n_classes = self.n_tasks * self.n_class_per_task
        self.order_list = cls_order[start * self.n_class_per_task: end * self.n_class_per_task]

        print("Create {} stream : {} tasks,  classes order {} ".format(self.split, self.n_tasks, self.order_list))
        print("Input shape (L, D): {}".format(tuple(input_size_match[data])))

    def load_data(self, load_subject=False):
        """
        Load data from .pkl files into np arrays.
        For methods using subject labels, set load_subject = True
        """

        x_train = load_pickle(self.path + 'x_train.pkl')
        y_train = load_pickle(self.path + 'state_train.pkl')
        x_test = load_pickle(self.path + 'x_test.pkl')
        y_test = load_pickle(self.path + 'state_test.pkl')

        if load_subject:
            sub_train = load_pickle(self.path + 'subject_label_train.pkl')
            sub_train = sub_train.squeeze()
        else:
            sub_train = None

        if n_tasks[self.data] < 5 and self.split == 'val':
            x_test = load_pickle(self.path + 'x_val.pkl')
            y_test = load_pickle(self.path + 'state_val.pkl')

        return x_train, y_train.squeeze(), x_test, y_test.squeeze(), sub_train

    def setup(self, cut=0.9, load_subject=False):
        """
        Arrange the data into tasks, according to the class order.
        Each task has a train set, val set (for earlystop) and test set
        If load_subject=True, train set contains subject labels.
        """

        x_train, y_train, x_test, y_test, sub_train = self.load_data(load_subject)

        class_idx = 0
        train_size, test_size, val_size = 0, 0, 0
        for t in range(self.n_tasks):
            classes_in_task_t = []  # class index that contains in task t

            for i in range(self.n_class_per_task):
                classes_in_task_t.append(self.order_list[class_idx])
                class_idx += 1

            if load_subject:
                x_train_t, y_train_t, sub_train_t = extract_samples_according_to_labels_with_sub(x_train, y_train, sub_train, classes_in_task_t)
            else:
                x_train_t, y_train_t = extract_samples_according_to_labels(x_train, y_train, classes_in_task_t)

            x_test_t, y_test_t = extract_samples_according_to_labels(x_test, y_test, classes_in_task_t)

            if self.scenario == 'class':
                # map the class labels to the ordered ones
                y_train_t = np.array([self.order_list.index(i) for i in y_train_t])
                y_test_t = np.array([self.order_list.index(i) for i in y_test_t])

            elif self.scenario == 'domain':
                # TODO: extend it to DIL in the future
                pass

            if load_subject:
                train_data = (x_train_t, y_train_t, sub_train_t)
                train_data, val_data = make_valid_from_train_with_sub(train_data, cut=cut)
            else:
                train_data = (x_train_t, y_train_t)
                train_data, val_data = make_valid_from_train(train_data, cut=cut)
            test_data = (x_test_t, y_test_t)

            train_size += train_data[0].shape[0]
            val_size += val_data[0].shape[0]
            test_size += test_data[0].shape[0]



            self.tasks.append((train_data, val_data, test_data))

        print("Training set size: {}; Val set size: {}; Test set size: {}".format(train_size, val_size, test_size))




    def setup_offline(self, cut=0.9):
        x_train, y_train, x_test, y_test, _ = self.load_data()
        x_train, y_train = extract_samples_according_to_labels(x_train, y_train, self.order_list)
        x_test, y_test = extract_samples_according_to_labels(x_test, y_test, self.order_list)

        # map the class labels to the ordered ones
        y_train = np.array([self.order_list.index(i) for i in y_train])
        y_test = np.array([self.order_list.index(i) for i in y_test])

        train_data = (x_train, y_train)
        test_data = (x_test, y_test)
        train_data, val_data = make_valid_from_train(train_data, cut)

        train_size = train_data[0].shape[0]
        val_size = val_data[0].shape[0]
        test_size = test_data[0].shape[0]
        print("Training set size: {}; Val set size: {}; Test set size: {}".format(train_size, val_size, test_size))

        return train_data, val_data, test_data
    
    def shuffle(self,random_seed):
        """
        Shuffle the order of the tasks in the task stream.
        """
        # Taskok előkészítése a ciklus előtt
        n_tasks_exp = len(self.tasks)
        new_tasks = []
        classes_in_each_task = []

        for task_i in range(n_tasks_exp):
            (x_train_current, y_train_current), (x_val_current, y_val_current), (x_test_current, y_test_current) = self.tasks[task_i]
            
            # Véletlenszerűen kiválasztunk 3/4 részt az x_train_current-ből
            n_samples_train = len(x_train_current)
            n_keep_train = int(n_samples_train * 0.75)  # 3/4 rész megtartása
            np.random.seed(random_seed)
            indices_train = np.random.permutation(n_samples_train)
            keep_indices_train = indices_train[:n_keep_train]
            remove_indices_train = indices_train[n_keep_train:]

            # Megtartott és eltávolított train minták szétválasztása
            x_train_keep = x_train_current[keep_indices_train]
            y_train_keep = y_train_current[keep_indices_train]
            x_train_remove = x_train_current[remove_indices_train]
            y_train_remove = y_train_current[remove_indices_train]

            # Véletlenszerűen kiválasztunk 3/4 részt az x_val_current-ből
            n_samples_val = len(x_val_current)
            n_keep_val = int(n_samples_val * 0.75)  # 3/4 rész megtartása
            np.random.seed(random_seed)
            indices_val = np.random.permutation(n_samples_val)
            keep_indices_val = indices_val[:n_keep_val]
            remove_indices_val = indices_val[n_keep_val:]

            # Megtartott és eltávolított validációs minták szétválasztása
            x_val_keep = x_val_current[keep_indices_val]
            y_val_keep = y_val_current[keep_indices_val]
            x_val_remove = x_val_current[remove_indices_val]
            y_val_remove = y_val_current[remove_indices_val]

            # Véletlenszerűen kiválasztunk 3/4 részt az x_test_current-ből
            n_samples_test = len(x_test_current)
            n_keep_test = int(n_samples_test * 0.75)  # 3/4 rész megtartása
            np.random.seed(random_seed)
            indices_test = np.random.permutation(n_samples_test)
            keep_indices_test = indices_test[:n_keep_test]
            remove_indices_test = indices_test[n_keep_test:]

            # Megtartott és eltávolított teszt minták szétválasztása
            x_test_keep = x_test_current[keep_indices_test]
            y_test_keep = y_test_current[keep_indices_test]
            x_test_remove = x_test_current[remove_indices_test]
            y_test_remove = y_test_current[remove_indices_test]

            # Az új task adathalmazának összeállítása
            if task_i == 0:
                # Task 1: Csak a megtartott minták
                x_train_new = x_train_keep
                y_train_new = y_train_keep
                x_val_new = x_val_keep
                y_val_new = y_val_keep
                x_test_new = x_test_keep
                y_test_new = y_test_keep
            else:
                # Task 2 és továbbiak: Hozzáadjuk az előző task eltávolított mintáit
                x_train_new = np.concatenate([x_train_keep, removed_train_samples[0]], axis=0)
                y_train_new = np.concatenate([y_train_keep, removed_train_samples[1]], axis=0)
                x_val_new = np.concatenate([x_val_keep, removed_val_samples[0]], axis=0)
                y_val_new = np.concatenate([y_val_keep, removed_val_samples[1]], axis=0)
                x_test_new = np.concatenate([x_test_keep, removed_test_samples[0]], axis=0)
                y_test_new = np.concatenate([y_test_keep, removed_test_samples[1]], axis=0)

            # Az eltávolított minták mentése a következő task számára
            removed_train_samples = (x_train_remove, y_train_remove)
            removed_val_samples = (x_val_remove, y_val_remove)
            removed_test_samples = (x_test_remove, y_test_remove)

            # Az új task hozzáadása a listához
            new_tasks.append(((x_train_new, y_train_new), (x_val_new, y_val_new), (x_test_new, y_test_new)))

            # Az aktuális task osztályainak tárolása
            classes_in_each_task.append(np.unique(y_train_current))
            print(f"Task {task_i} - Classes in train data: {np.unique(y_train_current)}")


        # A task_stream.tasks frissítése az új taskokkal
        self.tasks = new_tasks
        return classes_in_each_task


def make_valid_from_train(dataset, cut=0.9):
    x_t, y_t = dataset
    x_tr, y_tr, x_val, y_val = [], [], [], []
    for cls in set(y_t.tolist()):
        x_cls, y_cls = extract_samples_according_to_labels(x_t, y_t, [cls])
        perm = torch.randperm(len(x_cls))
        x_cls, y_cls = x_cls[perm], y_cls[perm]
        split = int(len(x_cls) * cut)
        x_tr_cls, y_tr_cls = x_cls[:split], y_cls[:split]
        x_val_cls, y_val_cls = x_cls[split:], y_cls[split:]

        x_tr.append(x_tr_cls)
        y_tr.append(y_tr_cls)
        x_val.append(x_val_cls)
        y_val.append(y_val_cls)

    x_tr = np.concatenate(x_tr)
    x_val = np.concatenate(x_val)
    y_tr = np.concatenate(y_tr)
    y_val = np.concatenate(y_val)
    return (x_tr, y_tr), (x_val, y_val)


def make_valid_from_train_with_sub(dataset, cut=0.9):
    x_t, y_t, sub_t = dataset
    x_tr, y_tr, x_val, y_val, sub_tr, sub_val = [], [], [], [], [], []
    for cls in set(y_t.tolist()):
        x_cls, y_cls, sub_cls = extract_samples_according_to_labels_with_sub(x_t, y_t, sub_t, [cls])
        perm = torch.randperm(len(x_cls))
        x_cls, y_cls, sub_cls = x_cls[perm], y_cls[perm], sub_cls[perm]
        split = int(len(x_cls) * cut)
        x_tr_cls, y_tr_cls, sub_tr_cls = x_cls[:split], y_cls[:split], sub_cls[:split]
        x_val_cls, y_val_cls, sub_val_cls = x_cls[split:], y_cls[split:], sub_cls[split:]

        x_tr.append(x_tr_cls)
        y_tr.append(y_tr_cls)
        sub_tr.append(sub_tr_cls)
        x_val.append(x_val_cls)
        y_val.append(y_val_cls)

    x_tr = np.concatenate(x_tr)
    x_val = np.concatenate(x_val)
    y_tr = np.concatenate(y_tr)
    y_val = np.concatenate(y_val)
    sub_tr = np.concatenate(sub_tr)

    return (x_tr, y_tr, sub_tr), (x_val, y_val)


def get_cls_order(data, fix_order=False):
    if fix_order:
        return preset_orders[data]
    else:
        all_classes = np.arange(n_classes[data])
        np.random.shuffle(all_classes)
        cls_order = list(all_classes)
        return cls_order
    

