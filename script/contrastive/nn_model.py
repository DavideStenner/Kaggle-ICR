import os
import glob
import torch
import json

import numpy as np
import pandas as pd
import seaborn as sns
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F


from torch import nn
from enum import Enum
from typing import Tuple, List
from collections import OrderedDict
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

from script.tabnet.tab_network import TabNet, TabNetNoEmbeddings
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
    def __init__(self, input_size, output_size, activation):
        super(FFLayer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.ff_layer = nn.Sequential(
            nn.Linear(self.input_size, self.output_size),
            nn.BatchNorm1d(self.output_size),
            activation(),
        )
    def forward(self, x):
        return self.ff_layer(x)

class TabnetLayer(nn.Module):
    def __init__(self, **kwargs):
        super(TabnetLayer, self).__init__()
        self.tabnet = TabNetNoEmbeddings(
            **kwargs
        )
    def forward(self, x):
        pred, _ = self.tabnet(x)
        return pred
    
class ContrastiveClassifier(pl.LightningModule):
    def __init__(self, config: dict, valid_dataset: DataLoader=None):
        super().__init__()
        
        self.valid_dataset = valid_dataset
        self.config = config
        self.init_pretraining_setup()

        self.embedding_size = config['embedding_size']
        self.pos_weight = None if config['pos_weight'] is None else torch.FloatTensor([config['pos_weight']])
        num_features = config['num_features']

        self.classification_head = nn.Sequential(
            FFLayer(self.embedding_size, 100, nn.GELU),
            FFLayer(100, 100, nn.GELU),
            nn.Linear(100, 1)
        )
        
        # self.contrastive_ff = nn.Sequential(
        #     TabnetLayer(
        #         input_dim=num_features,
        #         output_dim=self.embedding_size,
        #         n_a=64,
        #         n_d=64,
        #         virtual_batch_size=16,
        #         group_attention_matrix=None
        #     ),
        #     Normalize()
        # )


        self.embedding_layer = nn.Sequential(
            FFLayer(num_features, num_features, nn.GELU),
            FFLayer(num_features, self.embedding_size, nn.GELU),
            FFLayer(self.embedding_size, self.embedding_size, nn.GELU),
        )
        self.embedding_head = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            Normalize()
        )
        self.contrastive_ff = nn.Sequential(
            self.embedding_layer,
            self.embedding_head
        )

        self.normalize_input = nn.BatchNorm1d(num_features)

        self.step_outputs = {
            'train': [],
            'val': [],
            'test': []
        }
        self.save_hyperparameters()

    def init_pretraining_setup(self) -> None:
        self.lr = self.config['lr_pretraining']

        self.criterion = CosineSimilarityLoss()
        self.pretraining: bool = True
        self.inference_embedding: bool = False

    def init_classifier_setup(self) -> None:
        self.lr = self.config['lr']
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.comp_loss = competition_log_loss

        self.froze_layer()
        self.pretraining: bool = False

    def init_embedding_inference_setup(self) -> None:
        self.inference_embedding: bool = True

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

        if mode != 'train':
            self.pretraining_inspection()
            
            if not self.pretraining:
                #evaluate on all dataset

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
    
    def pretraining_inspection(self):
        if not self.pretraining:
            return
        else:
            if self.valid_dataset is None:
                return
            else:

                embeddings = []
                labels = []

                for input, label in self.valid_dataset:
                    input_ = input.to('cuda:0')

                    input_ = self.normalize_input(input_)
                    embedding = self.contrastive_ff(input_).cpu().numpy().tolist()
                    label = label.numpy().tolist()

                    embeddings += (embedding)
                    labels += label

                pca_ = PCA(n_components=2)
                embeddings = pca_.fit_transform(np.array(embeddings))

                results = pd.DataFrame(
                    {
                        'pca_1': embeddings[:, 0],
                        'pca_2': embeddings[:, 1],
                        'labels': labels
                    }
                )
                sns.scatterplot(data=results, x="pca_1", y="pca_2", hue="labels")
                plt.show()


    def forward(self, inputs):
        
        if self.pretraining:
            if self.inference_embedding:
                input = self.normalize_input(inputs['sample_1'])
                embedding = self.contrastive_ff(input)
                return embedding
            else:
                input_1 = self.normalize_input(inputs['sample_1'])
                input_2 = self.normalize_input(inputs['sample_2'])

                sample_1 = self.contrastive_ff(input_1)
                sample_2 = self.contrastive_ff(input_2)

                embedding = [
                    sample_1, sample_2
                ]
                return embedding
            
        #classification
        else:
            embedding = self.contrastive_ff(inputs)
            output = self.classification_head(embedding)        
            output = torch.flatten(output)
            return output
    
    def predict_step(self, batch: torch.tensor, batch_idx: int):
        if self.pretraining:
            pred = self.forward(batch)

        else:
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
    col_used = feature_list + [original_tgt_label, 'Alpha']

    c_1_data = data.loc[
        c_1_simulated, col_used
    ].reset_index(drop=True)

    c_2_data = data.loc[
        c_2_simulated, col_used
    ].reset_index(drop=True)
    
    target_ = get_target_score(
        original_tgt_label,
        c_1_data, c_2_data
    )
    
    return [c_1_data, c_2_data], target_

def get_target_score(
        tgt_label: str,
        c_1_data: pd.DataFrame, c_2_data: pd.DataFrame
    ) -> pd.Series:

    target_  = (
        (c_2_data['Alpha'] == c_1_data['Alpha'])
    ).astype(float)

    # target_.loc[
    #     (c_2_data[tgt_label] == c_1_data[tgt_label]) &
    #     (c_2_data['Alpha'] != c_1_data['Alpha'])
    # ] = 0.5

    return target_

def get_training_dataset_loader(
        config_model: dict, train: pd.DataFrame, 
        fold_: int, target_col: str, feature_list: list
    ):
    train_dataset_contrastive = get_contrastive_dataset(
        data=train, fold_=fold_, validation=False, 
        target_col=target_col, feature_list=feature_list,
    )
    valid_dataset_contrastive = get_contrastive_dataset(
        data=train, fold_=fold_, validation=True, 
        target_col=target_col, feature_list=feature_list,
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

def run_nn_contrastive_experiment(
        config_experiment: dict, config_model: dict, 
        feature_list: list, target_col: str,
    ) -> None:
    
    train = pd.read_pickle(
        os.path.join(config_experiment['PATH_DATA'], 'processed_data.pkl')
    )[feature_list + ['fold', target_col, 'Alpha']]

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

        w_0, w_1 = calc_log_loss_weight(
            train.loc[
                train['fold']==fold_, config_experiment['TARGET_COL']
            ].values
        )
        config_model[f'pos_weight'] = w_1/w_0

        print(f'\n\nStarting fold {fold_}\n\n\n')
        print('\n\nStarting Pretraining\n\n\n')
        (
            train_loader_contrastive, valid_loader_contrastive,
            train_loader_training, valid_loader_training
        ) = get_training_dataset_loader(
            config_model=config_model, train=train, 
            fold_=fold_, target_col=target_col, feature_list=feature_list
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
        
        print_dataset = valid_loader_training if config_model['print_pretraining'] else None

        model_ = ContrastiveClassifier(config=config_model, valid_dataset=print_dataset)

        contrastive_trainer.fit(model_, train_loader_contrastive, valid_loader_contrastive)
        model_.init_classifier_setup()

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
        classifier_trainer.fit(model_, train_loader_training, valid_loader_training)

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