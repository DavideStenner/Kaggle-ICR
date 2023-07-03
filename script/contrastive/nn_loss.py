import torch

import torch.nn.functional as F

from torch import nn
from enum import Enum

from typing import Callable

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

    def __init__(self, distance_metric=SiameseDistanceMetric.MANHATTAN, margin: float = 5, size_average:bool = True):
        super(ContrastiveLoss, self).__init__()

        self.distance_metric = distance_metric
        self.margin = margin
        self.size_average = size_average

    def forward(self, embeddings, labels):
        rep_anchor, rep_other = embeddings['sample_1'], embeddings['sample_2']
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
    def __init__(self, pos_weight: float=None):
        super(CosineSimilarityLoss, self).__init__()
        if pos_weight is None:
            self.use_weight = False
            self.loss_fct = nn.MSELoss()
        else:
            self.use_weight = True
            self.loss_fct = WeightedTensorMSELoss()

    def forward(self, embeddings, labels):
        output = torch.cosine_similarity(embeddings['sample_1'], embeddings['sample_2'])

        if self.use_weight:
            return self.loss_fct(output, labels.view(-1), embeddings['mask_1_target'])
        else:
            return self.loss_fct(output, labels.view(-1))

class HardCosineSimilarityLoss(nn.Module):
    """
    CosineSimilarityLoss expects, that the InputExamples consists of two texts and a float label.

    It computes the vectors u = model(input_text[0]) and v = model(input_text[1]) and measures the cosine-similarity between the two.
    By default, it minimizes the following loss: ||input_label - cos_score_transformation(cosine_sim(u,v))||_2.

    :param model: model
    :param loss_fct: Which pytorch loss function should be used to compare the cosine_similartiy(u,v) with the input_label? By default, MSE:  ||input_label - cosine_sim(u,v)||_2
    :param cos_score_transformation: The cos_score_transformation function is applied on top of cosine_similarity. By default, the identify function is used (i.e. no change).

    """
    def __init__(self, pos_weight: float=None):
        super(HardCosineSimilarityLoss, self).__init__()

        self.pos_weight = pos_weight

        self.loss_fct = nn.MSELoss()

        if pos_weight is None:
            self.use_weight = False
        else:
            self.use_weight = True

    def forward(self, embeddings, labels):
        similarity_matrix = torch.cosine_similarity(embeddings['sample_1'], embeddings['sample_2'])

        loss_ = self.top_worst_hard_loss(similarity_matrix, labels, embeddings['original_target'])
        # loss_ = self.online_hard_loss(similarity_matrix, labels)

        if self.use_weight:
            weight_ = ((self.pos_weight - 1) * embeddings['original_target']) + 1
            return loss_ * weight_
        else:
            return loss_

    def top_worst_hard_loss(self, similarity_matrix, labels, original_target, n: int = 4):
        #sort and find worst/best
        sorted_dist, sorted_index = torch.sort(similarity_matrix, stable=True)
        sorted_labels = labels[sorted_index]
        poss_num = n if original_target==0 else 5*n

        mask_0 = sorted_labels == 0
        mask_1 = sorted_labels == 1

        worst_negs = sorted_dist[mask_0][-poss_num:]
        worst_poss = sorted_dist[mask_1][:poss_num]

        hardest_example = torch.cat((worst_negs, worst_poss))
        labels_example = torch.cat(
            (
                sorted_labels[mask_0][:poss_num],
                sorted_labels[mask_1][:poss_num]
            )
        )
            
        loss_ = self.loss_fct(hardest_example, labels_example.view(-1))
        return loss_

    def online_hard_loss(self, similarity_matrix, labels, original_target):
        negs = similarity_matrix[labels == 0]
        poss = similarity_matrix[labels == 1]

        hard_negative = torch.logical_and(
            labels == 0, 
            similarity_matrix > (poss.min() if len(poss) > 1 else negs.mean())
        )
        hard_positive = torch.logical_and(
            labels == 1,
            similarity_matrix < (negs.max() if len(negs) > 1 else poss.mean())
        )
        #select hard positive and hard negative pairs
        mask_hard = torch.logical_or(hard_negative, hard_positive)

        distance_hard, label_hard = similarity_matrix[mask_hard], labels[mask_hard]
        loss_ = self.loss_fct(distance_hard, label_hard.view(-1))
        return loss_
   
class WeightedTensorMSELoss(nn.Module):
    def __init__(self):
        super(WeightedTensorMSELoss, self).__init__()

    def forward(self, output, labels, mask_1_target):
        return torch.sum(mask_1_target * ((output - labels) ** 2))/torch.sum(mask_1_target)