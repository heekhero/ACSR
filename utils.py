import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Sampler

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1, reduction='mean'):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.reduction = reduction

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss

class NShotTaskSamplerFromArray(Sampler):
    def __init__(self, dataset: torch.utils.data.Dataset, indices = None, rank=0):
        super(NShotTaskSamplerFromArray, self).__init__(dataset)

        self.indices = indices
        self.rank = rank

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        for index in self.indices:

            yield index

class NShotTaskSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 batch_size: int = 1):

        super(NShotTaskSampler, self).__init__(dataset)
        self.episodes = episodes
        self.dataset = dataset

        self.n = n
        self.k = k
        self.q = q

        self.batch_count = 0
        self.batch_size = batch_size

    def __len__(self):
        return self.episodes

    def __iter__(self):
        batch = []
        for _ in range(self.episodes):
            self.batch_count += 1

            selected_classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.n, replace=False)
            df = self.dataset.df[self.dataset.df['class_id'].isin(selected_classes)]

            support_k = {k: None for k in selected_classes}
            for k in selected_classes:
                # Select support examples
                support = df[df['class_id'] == k].sample(self.k)
                support_k[k] = support

                batch += list(support['id'])

            for k in selected_classes:
                query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(self.q)
                batch += list(query['id'])

            if self.batch_count == self.batch_size:
                yield np.stack(batch)
                self.batch_count = 0
                batch = []

@torch.no_grad()
def accuracy(y_pred, y):
    return float(torch.eq(y_pred.argmax(dim=-1), y).sum().item()) / float(y_pred.shape[0])


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.stack(tensors_gather, dim=0)
    return output




