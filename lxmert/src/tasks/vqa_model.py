# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

import counting
import torch


# Max length including <bos> and <eos>
# MAX_VQA_LENGTH = 20
# MAX_VQA_LENGTH = 30
MAX_VQA_LENGTH = 40

class VQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim

        objects = 10
        self.counter = counting.Counter(objects, already_sigmoided=True)
        
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim + objects + 1, hid_dim * 2),
            GeLU(),
            # BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent, get_cross_attention_probs=False):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x, attn_probs = self.lxrt_encoder(sent, (feat, pos), get_cross_attention_probs=get_cross_attention_probs)

        # attn_probs is of shape (bs, attention_heads, num_objects, MAX_VQA_LENGTH)
        # pick the first attention head and first token representation
        attn_probs = attn_probs[:, 0, :, 0]
        pos = pos.transpose(1, 2)
        c = self.counter(pos, attn_probs)

        # Fuse counting vector and the original x vector
        x = torch.cat((x, c), 1) # (bs, hid_dim + objects + 1)

        logit = self.logit_fc(x)

        return logit

