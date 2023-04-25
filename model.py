import math
import numpy as np
import random
import torch
import torch.nn as nn
from transformers import BartModel,BartConfig
from PianoBart import PianoBart
import pickle

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
        y = y.last_hidden_state[-1]
        # convert embeddings back to logits for prediction
        ys = []
        for i, etype in enumerate(self.e2w):
            ys.append(self.proj[i](y))           # (batch_size, seq_len, dict_size)
        return ys

#test
if __name__=='__main__':
    config=BartConfig(max_position_embeddings=32, d_model=48)
    with open('./Data/Octuple.pkl', 'rb') as f:
        e2w, w2e = pickle.load(f)
    piano_bart=PianoBart(config,e2w,w2e)
    piano_bart_lm=PianoBartLM(piano_bart)
    #print(piano_bart_lm)
    input_ids_encoder = torch.randint(1, 10, (2, 32, 8))
    input_ids_decoder = torch.randint(1, 10, (2, 32, 8))
    encoder_attention_mask = torch.zeros((2, 32))
    decoder_attention_mask = torch.zeros((2, 32))
    for j in range(2):
        encoder_attention_mask[j, 31] += 1
        decoder_attention_mask[j, 31] += 1
        decoder_attention_mask[j, 30] += 1
    output=piano_bart_lm(input_ids_encoder,input_ids_decoder,encoder_attention_mask,decoder_attention_mask)
    for temp in output:
        print(temp.size())