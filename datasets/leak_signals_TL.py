"""
Dataset loading for pipeline leakage detection with transfer learning.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm
import logging
import random


SIGNAL_SIZE = 1024
DATANAME = ['0', '1', '2', '3']
LABEL_CONDITION = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
LABEL_POS = list(range(12))
LABEL_CLS = list(range(4))


def get_source_files(root, source_indices):
    """Generate source domain training and validation files."""
    data = []
    lab_pos = []
    lab_cls = []

    random_numbers = random.sample(range(0, 12), 12)
    logging.info("source_pos:{}".format(random_numbers))

    for i in tqdm(random_numbers):
        path1 = os.path.join(root, DATANAME[source_indices[0]][0], LABEL_CONDITION[i])
        data1, lab_pos1, lab_cls1 = data_load(path1, LABEL_POS[i], LABEL_CLS[source_indices[0]])
        data += data1
        lab_pos += lab_pos1
        lab_cls += lab_cls1

    return [data, lab_pos, lab_cls]


def get_files(root, indices):
    """Generate training and testing dataset files."""
    data = []
    lab_pos = []
    lab_cls = []

    for idx in tqdm(range(12)):
        path1 = os.path.join(root, DATANAME[indices[0]][0], LABEL_CONDITION[idx])
        data1, lab_pos1, lab_cls1 = data_load(path1, LABEL_POS[idx], LABEL_CLS[indices[0]])
        data += data1
        lab_pos += lab_pos1
        lab_cls += lab_cls1

    return [data, lab_pos, lab_cls]


def data_load(filename, label_pos, label_cls):
    """Load data from CSV files."""
    data = []
    lab_pos = []
    lab_cls = []
    data_path = []

    file_list = os.listdir(filename)
    for file in file_list:
        finale_path = os.path.join(filename, file)
        data_path.append(finale_path)

    for path in data_path:
        df = pd.read_csv(path, header=0)
        data_temp = df.values[:, [1, 3, 5]][:1792].astype(float)
        data.append(data_temp)
        lab_pos.append(label_pos)
        lab_cls.append(label_cls)

    return data, lab_pos, lab_cls


class leak_signals_TL:
    """Dataset class for transfer learning on leak signals."""

    num_classes = 4
    inputchannel = 1

    def __init__(self, data_dir, transfer_task, normlizetype="0-1", num_classes=12):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1:4]
        self.normlizetype = normlizetype
        self.num_classes = num_classes

        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
            ])
        }

    def data_split(self, transfer_learning=True):
        """Split dataset into train/val sets."""
        if transfer_learning:
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({
                "data": list_data[0],
                "label_pos": list_data[1],
                "label_cls": list_data[2]
            })
            train_pd, val_pd = train_test_split(
                data_pd, test_size=0.2, random_state=40, stratify=data_pd["label_pos"]
            )
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            all_data_pd = []
            for target in self.target_N:
                list_data = get_files(self.data_dir, target)
                data_pd_temp = pd.DataFrame({
                    "data": list_data[0],
                    "label_pos": list_data[1],
                    "label_cls": list_data[2]
                })
                all_data_pd.append(data_pd_temp)

            data_pd_final = pd.concat(all_data_pd, ignore_index=True)
            target_val = dataset(list_data=data_pd_final, transform=self.data_transforms['val'])

            return source_train, source_val, target_val
        else:
            list_data = get_source_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({
                "data": list_data[0],
                "label_pos": list_data[1],
                "label_cls": list_data[2]
            })
            train_pd, val_pd = train_test_split(
                data_pd, test_size=0.2, random_state=40, stratify=data_pd["label_pos"]
            )
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            all_data_pd = []
            for target in self.target_N:
                list_data = get_files(self.data_dir, target)
                data_pd_temp = pd.DataFrame({
                    "data": list_data[0],
                    "label_pos": list_data[1],
                    "label_cls": list_data[2]
                })
                all_data_pd.append(data_pd_temp)

            data_pd_final = pd.concat(all_data_pd, ignore_index=True)
            target_val = dataset(list_data=data_pd_final, transform=self.data_transforms['val'])

            return source_train, source_val, target_val