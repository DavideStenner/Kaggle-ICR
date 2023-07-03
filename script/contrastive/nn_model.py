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
from typing import Tuple, Dict
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import adjusted_rand_score, f1_score
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
            # TabnetLayer(
            #     input_dim=num_features,
            #     output_dim=50,
            #     n_a=16,
            #     n_d=16,
            #     virtual_batch_size=8,
            #     group_attention_matrix=None
            # ),
            # nn.BatchNorm1d(50),
            # nn.GELU()
        )

        self.classification_head = nn.Sequential(
            FFLayer(100, 50, nn.GELU),
            nn.Linear(50, 1),
        )
        
        # self.embedding_layer = nn.Sequential(
        #     TabnetLayer(
        #         input_dim=num_features,
        #         output_dim=self.embedding_size,
        #         n_a=64,
        #         n_d=64,
        #         virtual_batch_size=16,
        #         group_attention_matrix=None
        #     ),
        #     nn.BatchNorm1d(self.embedding_size),
        #     Normalize()
        # )


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
