import torch

import numpy as np
import pandas as pd

from typing import List
from torch.utils.data import DataLoader,Dataset

from script.contrastive.augment_nn import get_smart_combination

class ICRClassificationDataset(Dataset):
    def __init__(self, 
                dataset: pd.DataFrame, feature_list: list, inference: bool,
                target_col_name: str
        ):
        self.inference = inference

        self.feature = dataset[feature_list].values

        if not inference:
            self.labels = dataset[target_col_name].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        inputs = torch.tensor(self.feature[item, :], dtype=torch.float)

        if self.inference:
            return inputs
        else:
            label = torch.tensor(self.labels[item], dtype=torch.float)
        
            return inputs, label

class ICRContrastiveDataset(Dataset):
    def __init__(self, 
                dataset: List[pd.DataFrame], feature_list: list, inference: bool,
                target: np.array, mask_1_target: np.array
        ):
        self.inference = inference
        self.num_features = len(feature_list)
        self.feature = [
            data[feature_list].values
            for data in dataset
        ]

        if not inference:
            self.labels = target
            self.mask_1_target = mask_1_target

    def __len__(self):
        return self.feature[0].shape[0]

    def __getitem__(self, item):
        inputs = {
            'sample_1': torch.tensor(self.feature[0][item, :], dtype=torch.float),
            'sample_2': torch.tensor(self.feature[1][item, :], dtype=torch.float),
            'mask_1_target': torch.tensor(self.mask_1_target[item], dtype=torch.float)
        }

        if self.inference:
            return inputs
        else:
            label = torch.tensor(self.labels[item], dtype=torch.float)

            return inputs, label

class ICRContrastiveBySampleDataset(Dataset):
    def __init__(self, 
            data: pd.DataFrame, feature_list: list, target_col: str,
            inference: bool, weight: float = 1
        ):
        self.inference = inference
        self.num_features = len(feature_list)
        self.feature = data[feature_list].values
        self.weight = weight
        self.target = data[target_col]
        self.number_rows = data.shape[0]

    def __len__(self):
        return self.number_rows

    def __getitem__(self, item):
        anchor = torch.tensor(self.feature[item, :], dtype=torch.float).repeat(self.number_rows-1)
        contrast = torch.tensor(np.delete(self.feature, [item], axis=0), dtype=torch.float)
		
        inputs = {
            'sample_1': anchor,
            'sample_2': contrast,
            'mask_1_target': torch.ones(self.number_rows-1, dtype=torch.float) * self.weight
        }

        if self.inference:
            return inputs
        else:
            label_anchor = torch.tensor(self.target[item], dtype=torch.float).repeat(self.number_rows-1)
            label_contrast = torch.tensor(
                np.delete(self.target, [item], axis=0), dtype=torch.float
            )

            label = torch.eq(label_anchor, label_contrast).type(torch.float)
            return inputs, label

def get_dataset(
        data: pd.DataFrame, fold_: int, validation: bool,
        target_col: str, feature_list: list
    ) -> Dataset:
    
    mask_fold = data['fold']==fold_ if validation else data['fold']!=fold_
    
    data = data[mask_fold].reset_index(drop=True)

    dataset = ICRClassificationDataset(
        dataset=data, feature_list=feature_list, 
        inference=False, target_col_name=target_col
    )
    return dataset

def get_hard_contrastive_dataset(
        data: pd.DataFrame, fold_: int, validation: bool,
        target_col: str, feature_list: list
    ):
    mask_fold = data['fold']==fold_ if validation else data['fold']!=fold_
    
    data = data[mask_fold].reset_index(drop=True)
    dataset = ICRContrastiveBySampleDataset(
        data=data, feature_list=feature_list, target_col=target_col, 
        inference=False
    )
    return dataset

def get_contrastive_dataset(
        data: pd.DataFrame, fold_: int, validation: bool,
        target_col: str, feature_list: list, batch_size: int
    ) -> Dataset:
    
    mask_fold = data['fold']==fold_ if validation else data['fold']!=fold_
    
    data = data[mask_fold].reset_index(drop=True)

    input_, target_, mask_1_target = contrastive_pipeline(
        data=data, feature_list=feature_list, 
        original_tgt_label=target_col, batch_size=batch_size
    )
    dataset = ICRContrastiveDataset(
        dataset=input_, feature_list=feature_list, 
        inference=False, target=target_, mask_1_target=mask_1_target
    )
    return dataset

def contrastive_pipeline(
        data: pd.DataFrame, feature_list: list,
        original_tgt_label: str, batch_size: int,
    ) -> List[np.array]:

    c_1_simulated, c_2_simulated = get_smart_combination(
        data=data, original_tgt_label=original_tgt_label
    )
    # get_all_combination_stratified(
    #     data=data, original_tgt_label=original_tgt_label, 
    #     batch_size=batch_size
    # )
    col_used = feature_list + [original_tgt_label, 'Alpha']

    c_1_data = data.loc[
        c_1_simulated, col_used
    ].reset_index(drop=True)

    c_2_data = data.loc[
        c_2_simulated, col_used
    ].reset_index(drop=True)
    
    target_, mask_1_target = get_target_score(
        original_tgt_label,
        c_1_data, c_2_data
    )
    
    return [c_1_data, c_2_data], target_, mask_1_target

def get_target_score(
        tgt_label: str,
        c_1_data: pd.DataFrame, c_2_data: pd.DataFrame
    ) -> pd.Series:

    target_  = (
        (c_2_data[tgt_label] == c_1_data[tgt_label])
    ).astype(float)

    mask_1_target = np.ones(target_.shape)
    mask_1_target[
        (c_2_data[tgt_label] == c_1_data[tgt_label]) &
        (c_2_data[tgt_label] == 1)
    ] = 21.97

    mask_1_target[
        (c_2_data[tgt_label] != c_1_data[tgt_label])
    ] = 1

    return target_, mask_1_target

def get_training_dataset_loader(
        config_model: dict, train: pd.DataFrame, 
        fold_: int, target_col: str, feature_list: list, batch_size: int
    ):
    train_dataset_contrastive = get_hard_contrastive_dataset(
        data=train, fold_=fold_, validation=False, 
        target_col=target_col, feature_list=feature_list,
        # batch_size=batch_size
    )
    valid_dataset_contrastive = get_hard_contrastive_dataset(
        data=train, fold_=fold_, validation=True, 
        target_col=target_col, feature_list=feature_list,
        # batch_size=batch_size
    )
    
    train_loader_contrastive = DataLoader(
        train_dataset_contrastive,
        batch_size=config_model['batch_size_pretraining'],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=config_model['num_workers']
    )
    
    valid_loader_contrastive = DataLoader(
        valid_dataset_contrastive,
        batch_size=config_model['batch_size_pretraining']*2,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=config_model['num_workers']
    )

    train_dataset = get_dataset(
        data=train, fold_=fold_, validation=False, 
        target_col=target_col, feature_list=feature_list,
    )
    valid_dataset = get_dataset(
        data=train, fold_=fold_, validation=True, 
        target_col=target_col, feature_list=feature_list,
    )

    train_loader_training = DataLoader(
        train_dataset,
        batch_size=config_model['batch_size'],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=config_model['num_workers']
    )
    
    valid_loader_training = DataLoader(
        valid_dataset,
        batch_size=config_model['batch_size']*2,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=config_model['num_workers']
    )

    return (
        train_loader_contrastive, valid_loader_contrastive,
        train_loader_training, valid_loader_training
    )

