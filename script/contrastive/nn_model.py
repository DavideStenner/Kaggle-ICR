import os
import glob
import torch
import json

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.nn.functional as F


from torch import nn
from enum import Enum
from typing import Tuple, List
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset

from script.loss import competition_log_loss, calc_log_loss_weight
from script.contrastive.augment_nn import get_all_combination_stratified


class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1-F.cosine_similarity(x, y)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
    two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.

    Further information: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    :param model: model
    :param distance_metric: Function that returns a distance between two embeddings. The class SiameseDistanceMetric contains pre-defined metrices that can be used
    :param margin: Negative samples (label == 0) should have a distance of at least the margin value.
    :param size_average: Average by the size of the mini-batch.

    """

    def __init__(self, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, margin: float = 0.5, size_average:bool = True):
        super(ContrastiveLoss, self).__init__()

        self.distance_metric = distance_metric
        self.margin = margin
        self.size_average = size_average

    def forward(self, embeddings_list, labels):
        rep_anchor, rep_other = embeddings_list
        distances = self.distance_metric(rep_anchor, rep_other)
        losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
        return losses.mean() if self.size_average else losses.sum()

class CosineSimilarityLoss(nn.Module):
    """
    CosineSimilarityLoss expects, that the InputExamples consists of two texts and a float label.

    It computes the vectors u = model(input_text[0]) and v = model(input_text[1]) and measures the cosine-similarity between the two.
    By default, it minimizes the following loss: ||input_label - cos_score_transformation(cosine_sim(u,v))||_2.

    :param model: model
    :param loss_fct: Which pytorch loss function should be used to compare the cosine_similartiy(u,v) with the input_label? By default, MSE:  ||input_label - cosine_sim(u,v)||_2
    :param cos_score_transformation: The cos_score_transformation function is applied on top of cosine_similarity. By default, the identify function is used (i.e. no change).

    """
    def __init__(self, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity()):
        super(CosineSimilarityLoss, self).__init__()
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation


    def forward(self, embeddings, labels):
        output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
        return self.loss_fct(output, labels.view(-1))

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
        self.normalize = F.normalize
        
    def forward(self, x):
        x = self.normalize(x)
        return x

class FFLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FFLayer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.ff_layer = nn.Sequential(
            nn.Linear(self.input_size, self.output_size),
            nn.GELU(),
            nn.BatchNorm1d(self.output_size),
        )
    def forward(self, x):
        return self.ff_layer(x)

class ContrastiveClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.init_pretraining_setup()

        self.embedding_size = config['embedding_size']
        self.pos_weight = None if config['pos_weight'] is None else torch.FloatTensor([config['pos_weight']])
        num_features = config['num_features']

        self.classification_head = nn.Sequential(
            nn.Linear(self.embedding_size, 1)
        )
        self.contrastive_ff = nn.Sequential(
            FFLayer(num_features, self.embedding_size//5),
            FFLayer(self.embedding_size//5, self.embedding_size//2),
            FFLayer(self.embedding_size//2, self.embedding_size//2),
            FFLayer(self.embedding_size//2, self.embedding_size),
            Normalize()
        )
        
        self.step_outputs = {
            'train': [],
            'val': [],
            'test': []
        }
        self.save_hyperparameters()

    def init_pretraining_setup(self) -> None:
        self.lr = self.config['lr_pretraining']

        self.criterion = ContrastiveLoss()
        self.pretraining: bool = True

    def init_classifier_setup(self) -> None:
        self.lr = self.config['lr']
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.comp_loss = competition_log_loss

        self.froze_layer()
        self.pretraining: bool = False

    def froze_layer(
            self,
        ) -> None:

        for _, param in self.contrastive_ff.named_parameters():
            param.requires_grad = False
        
    def __classifier_metric_step(self, pred: torch.tensor, labels: torch.tensor) -> dict:
        #can't calculate auc on a single batch.
        if self.trainer.sanity_checking:
            return {'comp_loss': 0.5}
        labels = (
            labels.numpy() if self.config['accelerator'] == 'cpu'
            else labels.cpu().numpy()
        )
        pred = (
            pred.numpy() if self.config['accelerator'] == 'cpu'
            else pred.cpu().numpy()
        )
        comp_loss_score = self.comp_loss(labels, pred)

        return {'comp_loss': comp_loss_score}
    
    def __loss_step(self, 
            pred: torch.tensor | List[torch.tensor], 
            labels: torch.tensor
        ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        loss = self.criterion(pred, labels)
        return loss, pred, labels
    
    def training_step(self, batch, batch_idx):
        input_, labels = batch

        pred = self.forward(input_)

        loss, _, _ = self.__loss_step(pred, labels)
        self.step_outputs['train'].append(
            {'loss': loss}
        )

        return loss

    def validation_step(self, batch, batch_idx):

        input_, labels = batch
        pred = self.forward(input_)

        loss, pred, labels = self.__loss_step(pred, labels)
        self.step_outputs['val'].append(
            {'loss': loss, 'pred': pred, 'labels': labels}
        )
        
    def test_step(self, batch, batch_idx):
        input_, labels = batch
        pred = self.forward(input_)

        loss, pred, labels = self.__loss_step(pred, labels)
        self.step_outputs['test'].append(
            {'loss': loss, 'pred': pred, 'labels': labels}
        )

    def on_training_epoch_end(self):
        self.__share_eval_res('train')
        
    def on_validation_epoch_end(self):
        self.__share_eval_res('val')

    def on_test_epoch_end(self):
        self.__share_eval_res('test')
    
    def __share_eval_res(self, mode: str):
        outputs = self.step_outputs[mode]
        loss = [out['loss'].reshape(1) for out in outputs]
        loss = torch.mean(torch.cat(loss))
        
        #initialize performance output
        res_dict = {
            f'{mode}_loss': loss
        }
        metric_message_list = [
            f'step: {self.trainer.global_step}',
            f'{mode}_loss: {loss:.5f}'
        ]
        if not self.pretraining:
            #evaluate on all dataset
            if mode != 'train':
                preds = [out['pred'] for out in outputs]
                preds = torch.sigmoid(torch.cat(preds))
                
                labels = [out['labels'] for out in outputs]
                labels = torch.cat(labels)
            
                metric_score = self.__classifier_metric_step(preds, labels)
                
                #calculate every metric on all batch
                metric_message_list += [
                    f'{mode}_{metric}: {metric_value:.5f}'
                    for metric, metric_value in metric_score.items()
                ]
                #get results
                res_dict.update(
                    {
                        f'{mode}_{metric}': metric_value
                        for metric, metric_value in metric_score.items()
                    }
                )
            else:
                pass

        if self.trainer.sanity_checking:
            pass
        else:
            print(', '.join(metric_message_list))
            self.log_dict(res_dict)
            
        #free memory
        self.step_outputs[mode].clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def forward(self, inputs):
        if self.pretraining:
            sample_1 = self.contrastive_ff(inputs['sample_1'])
            sample_2 = self.contrastive_ff(inputs['sample_2'])

            embedding = [
                sample_1, sample_2
            ]
            return embedding
        else:
            embedding = self.contrastive_ff(inputs)
            output = self.classification_head(embedding)        
            output = torch.flatten(output)
            return output
    
    def predict_step(self, batch: torch.tensor, batch_idx: int):
        assert not self.pretraining
        pred = self.forward(batch)
        pred = torch.flatten(pred)
        pred = torch.sigmoid(pred)
        
        return pred

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
                target: np.array
        ):
        self.inference = inference
        self.num_features = len(feature_list)
        self.feature = [
            data[feature_list].values
            for data in dataset
        ]

        if not inference:
            self.labels = target

    def __len__(self):
        return self.feature[0].shape[0]

    def __getitem__(self, item):
        inputs = {
            'sample_1': torch.tensor(self.feature[0][item, :], dtype=torch.float),
            'sample_2': torch.tensor(self.feature[1][item, :], dtype=torch.float)
        }

        if self.inference:
            return inputs
        else:
            label = torch.tensor(self.labels[item], dtype=torch.float)
        
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

def get_contrastive_dataset(
        data: pd.DataFrame, fold_: int, validation: bool,
        target_col: str, feature_list: list
    ) -> Dataset:
    
    mask_fold = data['fold']==fold_ if validation else data['fold']!=fold_
    
    data = data[mask_fold].reset_index(drop=True)

    input_, target_ = contrastive_pipeline(
        data=data, feature_list=feature_list, 
        original_tgt_label=target_col,
    )
    dataset = ICRContrastiveDataset(
        dataset=input_, feature_list=feature_list, 
        inference=False, target=target_
    )
    return dataset

def contrastive_pipeline(
        data: pd.DataFrame, feature_list: list,
        original_tgt_label: str
    ) -> List[np.array]:

    c_1_simulated, c_2_simulated = get_all_combination_stratified(data, original_tgt_label)
    col_used = feature_list + [original_tgt_label]

    c_1_data = data.loc[
        c_1_simulated, col_used
    ].reset_index(drop=True)

    c_2_data = data.loc[
        c_2_simulated, col_used
    ].reset_index(drop=True)

    target_  = (
        (c_2_data[original_tgt_label] == c_1_data[original_tgt_label])
    ).astype(int)

    return [c_1_data, c_2_data], target_

def run_nn_contrastive_experiment(
        config_experiment: dict, config_model: dict, 
        feature_list: list, target_col: str,
    ) -> None:
    
    train = pd.read_pickle(
        os.path.join(config_experiment['PATH_DATA'], 'processed_data.pkl')
    )[feature_list + ['fold', target_col]]

    train[feature_list] = (train[feature_list]-train[feature_list].mean())/train[feature_list].std()
    train[feature_list] = train[feature_list].fillna(0)
    for fold_ in range(config_experiment['N_FOLD']):

        log_folder = os.path.join(
            config_experiment['SAVE_RESULTS_PATH'],
            config_experiment['NAME'],
            'log',
            f'log_fold_{fold_}'
        )

        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        print(f'\n\nStarting fold {fold_}\n\n\n')
        print('\n\nStarting Pretraining\n\n\n')
        train_dataset = get_contrastive_dataset(
            data=train, fold_=fold_, validation=False, 
            target_col=target_col, feature_list=feature_list,
        )
        valid_dataset = get_contrastive_dataset(
            data=train, fold_=fold_, validation=True, 
            target_col=target_col, feature_list=feature_list,
        )
        w_0, w_1 = calc_log_loss_weight(
            train.loc[
                train['fold']==fold_, config_experiment['TARGET_COL']
            ].values
        )
        config_model[f'pos_weight'] = w_1/w_0
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config_model['batch_size_pretraining'],
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=config_model['num_workers']
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config_model['batch_size_pretraining']*2,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=config_model['num_workers']
        )

        loggers_pretraining = pl.loggers.CSVLogger(
            save_dir=log_folder,
            name='pretraining',
            version=config_model['version_experiment']
        )

        contrastive_trainer = pl.Trainer(
            max_epochs=config_model['max_epochs_pretraining'],
            max_steps=config_model['max_steps'],
            fast_dev_run=config_model['dev_run'], 
            accelerator=config_model['accelerator'],
            val_check_interval=config_model['val_check_interval_pretraining'],
            enable_progress_bar=(config_model['debug_run']) | (config_model['progress_bar']),
            num_sanity_val_steps=config_model['num_sanity_val_steps_pretraining'],
            logger=[loggers_pretraining],
            enable_checkpointing=False
        )
        
        model_ = ContrastiveClassifier(config=config_model)

        contrastive_trainer.fit(model_, train_loader, valid_loader)
        model_.init_classifier_setup()

        
        train_dataset = get_dataset(
            data=train, fold_=fold_, validation=False, 
            target_col=target_col, feature_list=feature_list,
        )
        valid_dataset = get_dataset(
            data=train, fold_=fold_, validation=True, 
            target_col=target_col, feature_list=feature_list,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config_model['batch_size'],
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=config_model['num_workers']
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config_model['batch_size']*2,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=config_model['num_workers']
        )

        loggers_training = pl.loggers.CSVLogger(
            save_dir=log_folder,
            name='training',
            version=config_model['version_experiment']
        )

        classifier_trainer = pl.Trainer(
            max_epochs=config_model['max_epochs'],
            max_steps=config_model['max_steps'],
            fast_dev_run=config_model['dev_run'], 
            accelerator=config_model['accelerator'],
            val_check_interval=config_model['val_check_interval'],
            enable_progress_bar=(config_model['debug_run']) | (config_model['progress_bar']),
            num_sanity_val_steps=config_model['num_sanity_val_steps'],
            logger=[loggers_training],
            enable_checkpointing=False
        )
        print('\n\nStarting training\n\n')
        classifier_trainer.fit(model_, train_loader, valid_loader)

def eval_nn_contrastive_experiment(
    config_experiment: dict, step: str, 
) -> None:
    assert step in ['training', 'pretraining']
    loss_name = 'val_loss' if step=='pretraining' else 'val_comp_loss'

    save_path = os.path.join(
        config_experiment['SAVE_RESULTS_PATH'],
        config_experiment['NAME']
    )
    path_results = os.path.join(
        save_path, 
        f'log\log_fold_*\{step}\*\metrics.csv'
    )
    metric_list = glob.glob(path_results)
    data = [
        pd.read_csv(path)
        for path in metric_list
    ]
    progress_dict = {
        'step': data[0]['step'],
    }

    progress_dict.update(
        {
            f"{loss_name}_fold_{i}": data[i][loss_name]
            for i in range(5)
        }
    )
    progress_df = pd.DataFrame(progress_dict)
    progress_df[f"average_{loss_name}"] = progress_df.loc[
        :, [loss_name in x for x in progress_df.columns]
    ].mean(axis =1)

    best_epoch = progress_df[f"average_{loss_name}"].argmin()
    best_step = progress_df.loc[
        best_epoch, "step"
    ]
    best_score = progress_df[f"average_{loss_name}"].min()

    best_score = {
        'best_epoch': int(best_epoch),
        'best_step': int(best_step),
        'best_score': best_score
    }
    print('\n')
    print(f'Best CV {loss_name} score for {step}')
    print(best_score)
    print('\n')
    
    with open(
            os.path.join(
                save_path,
                f'best_result_nn_{step}.txt'
            ), 'w'
        ) as file:
            json.dump(best_score, file)