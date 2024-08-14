#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Moussa Doumbouya (DROLabelSmoothingLoss)
# Copyright 2019 Shigeki Karita (Original LabelSmoothingLoss)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Label smoothing module with DRO"""

import torch
from torch import nn


class DROLabelSmoothingLoss(nn.Module):
    """Label-smoothing loss with Distributionally Robust Optimization

    :param int size: the number of class
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    :param int dro_group_count: Number of groups
    :param float dro_step_size: DRO step size
    :param float dro_q_epsilon: DRO epsillon (for numerical stability)
    """

    def __init__(
        self,
        size,
        padding_idx,
        smoothing,
        normalize_length=False,
        criterion=nn.KLDivLoss(reduction="none"),
        dro_group_count=0, dro_step_size=0.01, dro_q_epsilon=1e-10
    ):
        self._lsce_loss = LabelSmoothingLoss(
            size,
            padding_idx,
            smoothing,
            normalize_length=False,
            criterion=nn.KLDivLoss(reduction="none")
        )
        self.dro_group_count = dro_group_count
        self.dro_step_size = dro_step_size
        self.dro_q = torch.ones(self.dro_group_count) * 1.0/self.dro_group_count
        self.dro_q_epsilon = dro_q_epsilon


    def forward(self, x, target):
        """Compute loss between x and target.

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target:
            target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        group_loss = self._lsce_loss.forward(x, target)


        batch_lang_ids = [self.utt2category[_] for _ in utt_id] # TODO
        batch_lang_q_indices = []
        for lang_id in batch_lang_ids:
            if lang_id not in self.group_id_to_ix:
                self.group_id_to_ix[lang_id] = len(self.group_id_to_ix)
            batch_lang_q_indices.append(self.group_id_to_ix[lang_id])
        print(batch_lang_q_indices)

        for q_ix in set(batch_lang_q_indices): # unique set of groups in batch
            group_losses = torch.tensor([
                losses[i]/input_lengths[i] 
                for i in range(losses.shape[0])
                if batch_lang_q_indices[i] == q_ix
            ])

            group_mean_loss = torch.mean(group_losses)
            if self.use_running_mean:
                if len(self.mean_losses) == self.running_mean_window:
                    self.mean_losses.pop(0)
                self.mean_losses.append(group_mean_loss)
                self.dro_q[q_ix] *= torch.exp((group_mean_loss - sum(self.mean_losses)/len(self.mean_losses))* self.dro_step_size)
            else:
                self.dro_q[q_ix] *= torch.exp(group_mean_loss * self.dro_step_size)   

        self.normalize_dro_q()
        dro_losses = torch.stack([
            losses[ix] * self.dro_q[batch_lang_q_indices[ix]] * self.dro_group_count 
            for ix in range(losses.shape[0])
        ])


    def normalize_dro_q(self):
        # print("self.dro_q", self.dro_q)
        self.dro_q += self.dro_q_epsilon
        self.dro_q = self.dro_q / self.dro_q.sum()
        # print("self.dro_q", self.dro_q)
        print("normalized dro_q:")
        for group_id, group_ix in self.group_id_to_ix.items():
            print(f"q[group#{group_id}]= {self.dro_q[group_ix].item()}")
