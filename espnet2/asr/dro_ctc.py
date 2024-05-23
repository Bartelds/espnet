import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked
from torch import Tensor



class DROCTCLoss(torch.nn.Module):
    def __init__(self, blank=0, reduction='mean', zero_infinity=False, dro_group_count=0, dro_step_size=0.01):
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        self.dro_group_count = dro_group_count
        self.dro_step_size = dro_step_size

        # matches q in the algorithm
        self.dro_q = torch.ones(dro_group_count) * 1.0/dro_group_count
        self.group_id_to_ix = {}

    def forward(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor) -> Tensor:
        log_probs = torch.transpose(log_probs, 0, 1)

        batch_lang_ids = targets[:, 0] # TODO
        batch_lang_q_indices = []
        for lang_id in batch_lang_ids:
            lang_id = str(lang_id.item())
            if lang_id not in self.group_id_to_ix:
                self.group_id_to_ix[lang_id] = len(self.group_id_to_ix)
            batch_lang_q_indices.append(self.group_id_to_ix[lang_id])

        losses = F.ctc_loss(
            log_probs, 
            targets, input_lengths, target_lengths, 
            self.blank, reduction='none', # TODO: what is expected by the caller?
            zero_infinity=self.zero_infinity
        )

        for q_ix in set(batch_lang_q_indices): # unique set of groups in batch
            group_losses = torch.tensor([
                losses[i] 
                for i in range(losses.shape[0])
                if batch_lang_q_indices[i] == q_ix
            ])

            group_mean_loss = torch.mean(group_losses)
            self.dro_q[q_ix] *= torch.exp(group_mean_loss * self.dro_step_size)
        
        print(batch_lang_ids)
        print(batch_lang_q_indices)
        # normalize q
        self.normalize_dro_q()

        # print(dro_loss.shape)
        dro_loss = torch.stack([
            losses[ix] * self.dro_q[batch_lang_q_indices[ix]]
            for ix in range(losses.shape[0])
        ])
        return dro_loss


    def normalize_dro_q(self):
        self.dro_q = self.dro_q / self.dro_q.sum()
