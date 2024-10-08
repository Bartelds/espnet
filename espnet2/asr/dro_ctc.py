import logging
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked
from torch import Tensor
import pdb 

class DROCTCLoss(torch.nn.Module):
    def __init__(self, blank=0, reduction='mean', zero_infinity=False, dro_group_count=0, dro_step_size=0.01, dro_q_epsilon=1e-10, warmup_steps=0, use_running_mean=False,
    running_mean_window=-1, group_size_init=False):
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        self.dro_group_count = dro_group_count
        self.dro_step_size = dro_step_size

        self.dro_q = torch.ones(self.dro_group_count) * 1.0/self.dro_group_count
        self.dro_q_epsilon = dro_q_epsilon
        self.warmup_steps = warmup_steps
        self.use_running_mean = use_running_mean
        self.running_mean_window = running_mean_window
        self.group_size_init = group_size_init
        self.track_cnt = 0
        self.group_id_to_ix = {}

        if self.use_running_mean:
            self.mean_losses = []

    def init_weights(self, train_file, valid_file):
        if self.group_size_init:
            group_sizes = {}
            with open(str(train_file) + '/category2numbatches', 'r') as f:
                for line in f:
                    line = line.strip().split()
                    group_sizes[line[0]] = int(line[1])
                    group_size_init = [1/group_sizes[_] for _ in group_sizes]
                    self.group_id_to_ix = {key:idx for idx,key in enumerate(list(group_sizes.keys()))}
                    group_size_init_norm = [_/sum(group_size_init) for _ in group_size_init]
                    self.dro_q = torch.tensor(group_size_init_norm)
        
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

    def forward(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, utt_id: List[str]) -> Tensor:
        log_probs = torch.transpose(log_probs, 0, 1)

        batch_lang_ids = [self.utt2category[_] for _ in utt_id] # TODO
        batch_lang_q_indices = []
        for lang_id in batch_lang_ids:
            if lang_id not in self.group_id_to_ix:
                self.group_id_to_ix[lang_id] = len(self.group_id_to_ix)
            batch_lang_q_indices.append(self.group_id_to_ix[lang_id])
        print(batch_lang_q_indices)

        losses = F.ctc_loss(
            log_probs, 
            targets, input_lengths, target_lengths, 
            self.blank, reduction='none', # TODO: what is expected by the caller?
            zero_infinity=self.zero_infinity
        )

        # print stuff
        for i in range(len(losses)):
            lang_id = batch_lang_ids[i]
            loss_value = losses[i]
            input_length = input_lengths[i]
            target_length = target_lengths[i]
            print(f"Sample {i}: Language = {lang_id}, Loss = {loss_value}, Input Length = {input_length}, Target Length = {target_length}")

        if self.track_cnt > self.warmup_steps:
            for q_ix in set(batch_lang_q_indices): # unique set of groups in batch
                group_losses = torch.tensor([
                    losses[i]
                    # losses[i]/target_lengths[i] # changed from input_lengths[i]
                    for i in range(losses.shape[0])
                    if batch_lang_q_indices[i] == q_ix
                ])

                group_mean_loss = torch.sum(group_losses)
                if self.use_running_mean:
                    if len(self.mean_losses) == self.running_mean_window:
                        self.mean_losses.pop(0)
                    self.mean_losses.append(group_mean_loss)
                    self.dro_q[q_ix] *= torch.exp((group_mean_loss - sum(self.mean_losses)/len(self.mean_losses))* self.dro_step_size)
                else:
                    self.dro_q[q_ix] *= torch.exp(group_mean_loss * self.dro_step_size)   

        self.normalize_dro_q()
        dro_losses = torch.stack([
            losses[ix] * self.dro_q[batch_lang_q_indices[ix]] 
            for ix in range(losses.shape[0])
        ])

        self.track_cnt += 1
        if self.track_cnt > self.warmup_steps:
            return dro_losses
        else:
            return losses

    def normalize_dro_q(self):
        # print("self.dro_q", self.dro_q)
        self.dro_q += self.dro_q_epsilon
        self.dro_q = self.dro_q / self.dro_q.sum()
        # print("self.dro_q", self.dro_q)
        print("normalized dro_q:")
        for group_id, group_ix in self.group_id_to_ix.items():
            print(f"q[group#{group_id}]= {self.dro_q[group_ix].item()}")
