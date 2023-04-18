import math
import numpy as np
import random
import torch
import torch.nn as nn
from transformers import BartModel
from PianoBart import PianoBart


class PianoBartLM(nn.Module):
    def __init__(self, pianobart: PianoBart):
        super().__init__()
        self.pianobart = pianobart
        self.mask_lm = MLM(self.pianobart.e2w, self.pianobart.n_tokens, self.pianobart.hidden_size)

    def forward(self,input_ids_encoder, input_ids_decoder, encoder_attention_mask=None, decoder_attention_mask=None):
        x = self.pianobart(input_ids_encoder, input_ids_decoder, encoder_attention_mask, decoder_attention_mask)
        return self.mask_lm(x)


class MLM(nn.Module):
    def __init__(self, e2w, n_tokens, hidden_size):
        super().__init__()
        # proj: project embeddings to logits for prediction
        self.proj = []
        for i, etype in enumerate(e2w):
            self.proj.append(nn.Linear(hidden_size, n_tokens[i]))
        self.proj = nn.ModuleList(self.proj)  # 必须用这种方法才能像列表一样访问网络的每层
        self.e2w = e2w

    def forward(self, y):
        # feed to bart
        y = y.output_hidden_states[-1]
        # convert embeddings back to logits for prediction
        ys = []
        for i, etype in enumerate(self.e2w):
            ys.append(self.proj[i](y))           # (batch_size, seq_len, dict_size)
        return ys