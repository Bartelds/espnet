import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked
from torch import Tensor
import pdb 

class DROCTCLoss(torch.nn.Module):
    def __init__(self, blank=0, reduction='mean', zero_infinity=False, dro_group_count=0, dro_step_size=0.01, implementation="ananjan"):
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        self.dro_group_count = dro_group_count
        self.dro_step_size = dro_step_size

        # matches q in the algorithm
        self.implementation = implementation
        if self.implementation == "ananjan":
            nums = [1/756, 1/1406, 1/332, 1/316, 1/273, 1/217]
            nums_norm = [_/sum(nums) for _ in nums]
            self.dro_q = torch.tensor(nums_norm)
            self.track = 0
            self.limit = 0
            self.mean_losses = [] # maintain running mean of losses
            self.window = 50
        else:
            self.dro_q = torch.ones(self.dro_group_count) * 1.0/self.dro_group_count 
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

        if self.implementation == "ananjan":
            if self.track > self.limit:
                for q_ix in set(batch_lang_q_indices): # unique set of groups in batch
                    group_losses = torch.tensor([
                        losses[i]/input_lengths[i] 
                        for i in range(losses.shape[0])
                        if batch_lang_q_indices[i] == q_ix
                    ])

                    group_mean_loss = torch.mean(group_losses)
                    if len(self.mean_losses) == self.window:
                        self.mean_losses.pop(0)
                    self.mean_losses.append(group_mean_loss)
                    self.dro_q[q_ix] *= torch.exp((group_mean_loss - sum(self.mean_losses)/len(self.mean_losses))* self.dro_step_size)
        else:
            for q_ix in set(batch_lang_q_indices): # unique set of groups in batch
                group_losses = torch.tensor([
                    losses[i] 
                    for i in range(losses.shape[0])
                    if batch_lang_q_indices[i] == q_ix
                ])

                group_mean_loss = torch.mean(group_losses)
                self.dro_q[q_ix] *= torch.exp(group_mean_loss * self.dro_step_size)

        self.normalize_dro_q()
        dro_losses = torch.stack([
            losses[ix] * self.dro_q[batch_lang_q_indices[ix]] * self.dro_group_count 
            for ix in range(losses.shape[0])
        ])

        if self.implementation == "ananjan":
            self.track += 1
            if self.track > self.limit:
                return dro_losses
            else:
                return losses
        else:
            return dro_losses

    def normalize_dro_q(self):
        # print("self.dro_q", self.dro_q)
        if self.implementation == "ananjan":
            self.dro_q += 0.001
        else:
            self.dro_q += 1e-10
        self.dro_q = self.dro_q / self.dro_q.sum()
        # print("self.dro_q", self.dro_q)
        print("normalized dro_q:")
        for group_id, group_ix in self.group_id_to_ix.items():
            print(f"q[group#{group_id}]= {self.dro_q[group_ix].item()}")