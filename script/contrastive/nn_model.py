import os
import torch
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import nn
from typing import Tuple, Dict, Optional
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import adjusted_rand_score, f1_score, log_loss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from script.loss import competition_log_loss
from script.tabnet.tab_network import TabNetNoEmbeddings
from script.contrastive.nn_loss import HardCosineSimilarityLoss, CosineSimilarityLoss, ContrastiveLoss

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
            nn.Linear(self.input_size, self.output_size, bias=False),
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

        self.embedding_size = config['embedding_size']
        self.pos_weight = None if config['pos_weight'] is None else torch.FloatTensor([config['pos_weight']])
        self.num_features = config['num_features']
        self.init_pretraining_setup()

        self.embedding_ff = FFLayer(self.embedding_size, 50, nn.GELU)
        self.feature_ff = nn.Sequential(
            FFLayer(self.num_features, 50, nn.GELU),
            FFLayer(50, 50, nn.GELU)
        )

        self.classification_head = nn.Sequential(
            FFLayer(100, 50, nn.GELU),
            nn.Linear(50, 1),
        )
        
        self.embedding_layer = nn.Sequential(
            FFLayer(self.num_features, self.num_features, nn.GELU),
            FFLayer(self.num_features, self.embedding_size, nn.GELU),
            FFLayer(self.embedding_size, self.embedding_size, nn.GELU),
        )
        self.embedding_head = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size, bias=False),
            nn.BatchNorm1d(self.embedding_size),
        )

        self.contrastive_ff = nn.Sequential(
            self.embedding_layer,
            self.embedding_head,
            Normalize()
        )
        self.get_embedding = nn.Sequential(
            self.embedding_layer,
            self.embedding_head,
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

    def unfreeze_layer(
            self,
    ) -> None:
        
        for _, param in self.contrastive_ff.named_parameters():
            param.requires_grad = True

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
            pred: torch.tensor | Dict, 
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
        
        # if (not self.pretraining) & (self.current_epoch == 1):
        #     self.unfreeze_layer()
        
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
            metric_message_list = self.pretraining_inspection(metric_message_list, mode)
            
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
        # scheduler = CosineAnnealingLR(optimizer,
        #     T_max=10,
        #     eta_min=1e-5,
        # )

        # optimizer_dict = dict(
        #     optimizer=optimizer,
        #     lr_scheduler=scheduler
        # )
        return optimizer
    
    def euclidean_distance(self, v1, v2):
        return np.sqrt(np.sum((v1 - v2) ** 2)) 

    def manhattan_distance(self, v1, v2):
        return np.sum(np.abs(v1 - v2))
    
    def get_nearest(self, center_label: np.array, embeddings: np.array):
        label_ = [
            np.argmin(
                [
                    self.manhattan_distance(
                        embeddings[row, :], center_label[lab, :]
                    )
                    for lab in range(center_label.shape[0])
                ]
            )
            for row in range(embeddings.shape[0])
        ]
        return label_            

    def pretraining_inspection(self, metric_message_list:list, mode: str):
        if not self.pretraining:
            return metric_message_list
        else:
            if self.valid_dataset is None:
                return metric_message_list
            else:
                self.eval()

                embeddings = []
                labels = []

                for input_, label in self.valid_dataset:
                    if torch.cuda.is_available():
                        input_ = input_.to('cuda:0')

                    embedding = self.contrastive_ff(input_).cpu().numpy().tolist()
                    label = label.numpy().tolist()

                    embeddings += embedding
                    labels += label

                embeddings = np.array(embeddings)
                labels = np.array(labels)
                
                pca_ = PCA(n_components=2)
                embeddings_pca = pca_.fit_transform(embeddings)

                results = pd.DataFrame(
                    {
                        'pca_1': embeddings_pca[:, 0],
                        'pca_2': embeddings_pca[:, 1],
                        'labels': labels
                    }
                )
                for dataset, name in [(embeddings_pca, 'pca'), (embeddings, 'all')]:
                    center_label = np.stack(
                        [
                            np.median(dataset[labels==x, :], axis=0)
                            for x in range(2)
                        ]
                    )
                    if name == 'pca':
                        center_label_pca = center_label
                        
                    label_ = self.get_nearest(center_label, dataset)
                    
                    metric_score = adjusted_rand_score(labels, label_)
                    metric_message_list.append(
                        f'{mode}_adj_rand_{name}: {metric_score:.5f}'
                    )
                    f1_metric_score = f1_score(labels, label_)
                    metric_message_list.append(
                        f'{mode}_f1_{name}: {f1_metric_score:.5f}'
                    )

                if self.config['print_pretraining']:
                    plot = sns.scatterplot(data=results, x="pca_1", y="pca_2", hue="labels").get_figure()

                    for label_center, center in enumerate(center_label_pca):
                        plt.scatter(
                            center[0], center[1], marker="D", s=20, color="red"
                        )

                    plot.savefig(
                        os.path.join(
                            self.config['plot_folder'],
                            f'{self.trainer.global_step}_cluster.png'
                        )
                    )

                    if self.config['show_plot']:
                        plot
                        plt.show()
                    plt.close(plot)


                self.train()

                return metric_message_list

    def forward(self, inputs):
        
        if self.pretraining:
            if self.inference_embedding:
                embedding = self.contrastive_ff(inputs['sample_1'])
                return embedding
            else:

                sample_1 = self.contrastive_ff(inputs['sample_1'])
                sample_2 = self.contrastive_ff(inputs['sample_2'])

                embedding = {
                    'sample_1': sample_1,
                    'sample_2': sample_2,
                    'original_target': inputs['original_target']
                }
                return embedding
            
        #classification
        else:

            embedding = self.get_embedding(inputs)
            embedding = self.embedding_ff(embedding)
        
            feature = self.feature_ff(inputs)

            all_feature = torch.concat((embedding, feature),dim=-1)

            output = self.classification_head(all_feature)        
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


    

class BinaryConstrastiveHead(nn.Module):
    """
    This loss was used in our SBERT publication (https://arxiv.org/abs/1908.10084) to train the SentenceTransformer
    model on NLI data. It adds a softmax classifier on top of the output of two transformer networks.

    :param concatenation_sent_rep: Concatenate vectors u,v for the softmax classifier?
    :param concatenation_sent_difference: Add abs(u-v) for the softmax classifier?
    :param concatenation_sent_multiplication: Add u*v for the softmax classifier?
    :param loss_fct: Optional: Custom pytorch loss function. If not set, uses nn.CrossEntropyLoss()

    """
    def __init__(self,
            embedding_dimension: int,
            concatenation_sent_rep: bool = True,
            concatenation_sent_difference: bool = True,
            concatenation_sent_multiplication: bool = True
        ):
        super(BinaryConstrastiveHead, self).__init__()
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1

        self.classifier = nn.Linear(num_vectors_concatenated * embedding_dimension, 1)

    def forward(self, embedding_1, embedding_2):

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(embedding_1)
            vectors_concat.append(embedding_2)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(embedding_1 - embedding_2))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(embedding_1 * embedding_2)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        output = torch.flatten(output)

        return output


class ContrastiveClassifierProb(pl.LightningModule):
        def __init__(self, config: dict, train_dataset: DataLoader, valid_dataset: DataLoader):
            super().__init__()
            
            self.valid_dataset = valid_dataset
            self.train_dataset = train_dataset
            self.config = config

            self.embedding_size = config['embedding_size']
            self.num_features = config['num_features']
            self.init_pretraining_setup()
            
            self.embedding_layer = nn.Sequential(
                FFLayer(self.num_features, self.num_features, nn.GELU),
                FFLayer(self.num_features, self.embedding_size, nn.GELU),
                FFLayer(self.embedding_size, self.embedding_size, nn.GELU),
            )
            self.embedding_head = nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size, bias=False),
                nn.BatchNorm1d(self.embedding_size),
            )

            self.contrastive_ff = nn.Sequential(
                self.embedding_layer,
                self.embedding_head,
                Normalize()
            )
            self.binary_head_contrastive = BinaryConstrastiveHead(embedding_dimension=self.embedding_size)

            self.step_outputs = {
                'train': [],
                'val': [],
                'test': []
            }
            self.save_hyperparameters()
        
        def init_pretraining_setup(self) -> None:
            self.lr = self.config['lr_pretraining']
            self.weight = 1
            self.weight_tensor = torch.tensor([self.weight-1], dtype=torch.float)

            if torch.cuda.is_available():
                self.weight_tensor = torch.tensor([self.weight-1], dtype=torch.float).to('cuda')

            self.criterion = nn.BCEWithLogitsLoss()#reduction='none')
            self.comp_loss = competition_log_loss

        def __loss_step(self, 
                pred: torch.tensor | Dict, labels: torch.tensor,
                inputs: torch.tensor | Dict,
            ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
            loss = self.criterion(pred, labels)
            # rescale_weight = ((inputs['original_target_flattened'] * self.weight_tensor) + 1)
            # loss = (loss * rescale_weight).mean()

            return loss, pred, labels
        
        def training_step(self, batch, batch_idx):
            input_, labels = batch

            pred = self.forward(input_)

            loss, _, _ = self.__loss_step(pred, labels, input_)
            self.step_outputs['train'].append(
                {'loss': loss}
            )

            return loss

        def validation_step(self, batch, batch_idx):

            input_, labels = batch
            pred = self.forward(input_)

            loss, pred, labels = self.__loss_step(pred, labels, input_)
            self.step_outputs['val'].append(
                {'loss': loss, 'pred': pred, 'labels': labels}
            )
            
        def test_step(self, batch, batch_idx):
            input_, labels = batch
            pred = self.forward(input_)

            loss, pred, labels = self.__loss_step(pred, labels, input_)
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
            if self.trainer.sanity_checking:
                pass
            else:
                if mode != 'train':
                    metric_score, df_metric = self.retrieval_training_inspection()
                    
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
                    print(f'step: {self.trainer.global_step}, {mode}_loss: {loss:.5f}\n')
                    print(df_metric.to_markdown())

                    self.log_dict(res_dict)
                else:
                    pass
                
            #free memory
            self.step_outputs[mode].clear()
        
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer

        def contrastive_train_prediction(self, 
                train_loader: DataLoader, valid_loader: DataLoader,
                inference: bool = False
            ):
            """method used to create vector of prediction between train and valid loader. 
            Final prediction with top-k is done on retrieval_top_k_prediction

            Args:
                train_loader (DataLoader): 
                valid_loader (DataLoader): 
                inference (bool, optional):Defaults to False.
            """
            self.eval()

            train_embedding = []
            valid_embeddings = []

            train_labels = []
            valid_labels = []

            if inference:
                if torch.cuda.is_available():
                    self.contrastive_ff = self.contrastive_ff.to('cuda:0')
                    self.binary_head_contrastive = self.binary_head_contrastive.to('cuda:0')

            for input_train, label_train in train_loader:
                if torch.cuda.is_available():
                    input_train = input_train.to('cuda:0')
                    label_train = label_train.to('cuda:0')
                    
                train_labels.append(label_train)    
                train_embedding.append(self.contrastive_ff(input_train))

            for input_validation in valid_loader:
                if not inference:
                    input_validation, label_validation =  input_validation

                    valid_labels.append(label_validation)
                    
                if torch.cuda.is_available():
                    input_validation = input_validation.to('cuda:0')

                valid_embeddings.append(self.contrastive_ff(input_validation))

            train_labels, train_embedding = torch.concat(train_labels), torch.concat(train_embedding)

            valid_embeddings = torch.concat(valid_embeddings)

            size_train = train_embedding.shape[0]
            
            pred_list = []
            for valid_item in range(valid_embeddings.shape[0]):
                valid_chunk = valid_embeddings[valid_item, :].repeat(size_train, 1)

                pred_ = torch.sigmoid(self.binary_head_contrastive(train_embedding, valid_chunk))
                
                pred_list.append(pred_)

            if inference:
                return pred_list, train_labels
            else:
                valid_labels = torch.concat(valid_labels)
                return pred_list, train_labels, valid_labels

        def retrieval_top_k_prediction(self,
            train_loader: DataLoader, valid_loader: DataLoader, top_k: int,
            inference: bool = False
        ):
            """method to calculate top-k prediction. used during inference. Single pipeline which apply contrastive prediction and top-k retrieval.

            Args:
                train_loader (DataLoader): 
                valid_loader (DataLoader): 
                top_k (int): _description_
                inference (bool, optional): Defaults to False.
            """
            if inference:
                pred_list, train_labels = self.contrastive_train_prediction(train_loader, valid_loader, inference)
            else:
                pred_list, train_labels, valid_labels = self.contrastive_train_prediction(train_loader, valid_loader, inference)
            
            pred_array = self.retrieval_top_k_ensemble(pred_list, train_labels, top_k)

            if inference:
                return pred_array.detach()
            else:
                return pred_array, valid_labels
        
        def retrieval_top_k_ensemble(self,
            pred_list: list, train_labels: list, top_k: int
        ):
            """method to apply ensemble on top k prediction

            Args:
                pred_list (list):
                train_labels (list): 
                top_k (int): s
            """
            pred_array = []

            for pred_ in pred_list:
                top_k_pred = self.get_top_k(pred_, train_labels, top_k)
                pred_array.append(top_k_pred)

            pred_array = torch.concat(pred_array)
            return pred_array
        
        def get_top_k(self, pred_: torch.tensor, train_labels: torch.tensor, top_k: int) -> torch.tensor:
            """calculate top k given a vector

            Args:
                pred_ (torch.tensor):
                train_labels (torch.tensor):
                top_k (int):

            Returns:
                torch.tensor
            """
            ordered_pred, _ = torch.sort(pred_, descending=True)
            
            top_k_pred = ordered_pred[:top_k]
            # top_k_labels = train_labels[ordered_index[:top_k]]
            pred_ = top_k_pred

            # pred_ =  (1-top_k_pred) * (1- top_k_labels) + top_k_pred * top_k_labels

            return pred_.mean().reshape(1)
        
        def retrieval_training_inspection(self, top_k_list: list = [1, 5, 10, 15, 20, 50, 100, 200, 1000]):
            metric_dict = {}
            df_metric = []
            
            #calculate one time embedding to speed up
            pred_list, train_labels, label_validation = self.contrastive_train_prediction(self.train_dataset, self.valid_dataset)

            label_validation = label_validation.numpy()

            for top_k in top_k_list:
                pred_array = self.retrieval_top_k_ensemble(pred_list, train_labels, top_k)

                pred_array = (
                    pred_array.numpy() if self.config['accelerator'] == 'cpu'
                    else pred_array.cpu().numpy()
                )
                metric_score, metric_score_0, metric_score_1 = self.comp_loss(label_validation, pred_array, True)
                binary_ce = log_loss(label_validation, pred_array)

                result_df = pd.DataFrame(
                    {
                        'top_k': [top_k],
                        'comp_loss': [metric_score],
                        'binary_ce_0': [metric_score_0],
                        'binary_ce_1': [metric_score_1],
                        'binary_ce': [binary_ce]
                    }
                )

                df_metric.append(result_df)

                metric_dict.update(
                    {
                        f'comp_loss_{top_k}': metric_score,
                        f'binary_ce_0_{top_k}': metric_score_0,
                        f'binary_ce_1_{top_k}': metric_score_1,
                        f'binary_ce_{top_k}': binary_ce
                    }
                )

            df_metric = pd.concat(df_metric, axis=0, ignore_index=True)

            self.train()

            return metric_dict, df_metric

        def forward(self, inputs):
            sample_1 = self.contrastive_ff(inputs['sample_1'])
            sample_2 = self.contrastive_ff(inputs['sample_2'])
            output = self.binary_head_contrastive(sample_1, sample_2)
            return output
        
        def predict_step(self, batch: torch.tensor, batch_idx: int):
            pred = self.contrastive_ff(batch)
            pred = torch.sigmoid(pred)
            
            return pred
