import logging
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked
from torch import Tensor
import pdb 

class DROCTCLoss(torch.nn.Module):
    def __init__(self, blank=0, reduction='mean', zero_infinity=False, dro_group_count=0, dro_step_size=0.01, dro_q_epsilon=1e-10,
    accumulation=False, smoothing=0, agg="sum", normalize_grad=True):
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        self.dro_group_count = dro_group_count
        self.dro_step_size = dro_step_size

        self.dro_q = torch.ones(self.dro_group_count) * 1.0/self.dro_group_count
        self.dro_q_epsilon = dro_q_epsilon
        self.group_id_to_ix = {}
        self.agg = agg
        self.normalize_grad = normalize_grad

        self.accumulation = accumulation
        self.smoothing = smoothing

    def init_weights(self, train_file, valid_file):
        group_sizes = {}

        with open(str(train_file) + '/category2numbatches', 'r') as f:
            for line in f:
                line = line.strip().split()
                group_sizes[line[0]] = int(line[1])
        
        self.utt2category = {}
        with open(str(train_file) + '/utt2category', 'r') as f:
            for line in f:
                line = line.strip().split()
                self.utt2category[line[0]] = line[1]

        # Also load mappings for test and dev
        with open(str(valid_file) + '/utt2category', 'r') as f:
            for line in f:
                line = line.strip().split()
                self.utt2category[line[0]] = line[1]

        if self.accumulation:
            self.group_losses = {}
            for _ in range(len(group_sizes)):
                self.group_losses[_] = []

    def forward(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, utt_id: List[str], valid: bool = True) -> Tensor:
        log_probs = torch.transpose(log_probs, 0, 1)

        batch_lang_ids = [self.utt2category[_] for _ in utt_id]

        batch_lang_q_indices = []
        for lang_id in batch_lang_ids:
            if lang_id not in self.group_id_to_ix:
                self.group_id_to_ix[lang_id] = len(self.group_id_to_ix)
            batch_lang_q_indices.append(self.group_id_to_ix[lang_id])

        losses = F.ctc_loss(
            log_probs, 
            targets, input_lengths, target_lengths, 
            self.blank, reduction='none',
            zero_infinity=self.zero_infinity
        )

        # print stuff
        for i in range(len(losses)):
            lang_id = batch_lang_ids[i]
            filename = utt_id[i]
            loss_value = losses[i]
            input_length = input_lengths[i]
            target_length = target_lengths[i]
            if valid:
                print(f"Validation Sample {i}: Language = {lang_id}, Filename = {filename}, Loss = {loss_value}, Input Length = {input_length}, Target Length = {target_length}")
            else:
                print(f"Training Sample {i}: Language = {lang_id}, Filename = {filename}, Loss = {loss_value}, Input Length = {input_length}, Target Length = {target_length}")

        step_size = self.dro_step_size

        if not valid:
            for q_ix in set(batch_lang_q_indices): # unique set of groups in batch
                group_losses = torch.tensor([
                    losses[i]
                    for i in range(losses.shape[0])
                    if batch_lang_q_indices[i] == q_ix
                ])

                if (self.agg == "sum"):
                    group_mean_loss = torch.sum(group_losses)
                else:
                    group_mean_loss = torch.mean(group_losses)

                if not self.accumulation:
                    if self.smoothing > 0:
                        # add the smoothing hyperparameter
                        self.dro_q[q_ix] *= torch.exp((group_mean_loss * step_size) / (self.dro_q[q_ix] + self.smoothing))
                        print("Update Magnitude", torch.exp((group_mean_loss * step_size) / (self.dro_q[q_ix] + self.smoothing)))
                    else:
                        self.dro_q[q_ix] *= torch.exp(group_mean_loss * step_size) 
                        print("Update Magnitude", torch.exp(group_mean_loss * step_size))
                else:
                    print("Loss Stored")
                    self.group_losses[q_ix].append(group_mean_loss)

            if self.accumulation:
                check = True
                for _ in self.group_losses:
                    if len(self.group_losses[_]) == 0:
                        check = False
                        break 

                if check:
                    for _ in self.group_losses:
                        update_term = sum(self.group_losses[_])/len(self.group_losses[_])
                        if self.smoothing > 0:
                            self.dro_q[_] *= torch.exp((update_term * step_size)/(self.dro_q[_] + self.smoothing))
                            print("Update Magnitude", torch.exp((update_term * step_size)/(self.dro_q[_] + self.smoothing)))
                        else:
                            self.dro_q[_] *= torch.exp(update_term * step_size)
                            print("Update Magnitude", torch.exp(update_term * step_size))

                    self.normalize_dro_q()
                    for _ in self.group_losses:
                        self.group_losses[_] = []
            else:
                self.normalize_dro_q()
        
        if self.normalize_grad:
            # multiply loss by number of groups
            dro_losses = torch.stack([
                losses[ix] * self.dro_q[batch_lang_q_indices[ix]] 
                * self.dro_group_count
                for ix in range(losses.shape[0])
            ])
        else:
            dro_losses = torch.stack([
                losses[ix] * self.dro_q[batch_lang_q_indices[ix]] 
                for ix in range(losses.shape[0])
            ])
        
        if not valid:
            return dro_losses
        else:
            return losses

    def normalize_dro_q(self):
        self.dro_q += self.dro_q_epsilon # to prevent zero weights
        self.dro_q = self.dro_q / self.dro_q.sum()
        print("normalized dro_q:")
        for group_id, group_ix in self.group_id_to_ix.items():
            print(f"q[group#{group_id}]= {self.dro_q[group_ix].item()}")
