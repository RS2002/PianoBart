import math
import numpy as np
import random

import torch
import torch.nn as nn
from transformers import BertModel,BertConfig
import pickle


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# BERT model: similar approach to "felix"
class MidiBert(nn.Module):
    def __init__(self, bertConfig, e2w, w2e):
        super().__init__()

        self.bert = BertModel(bertConfig)
        bertConfig.d_model = bertConfig.hidden_size
        self.hidden_size = bertConfig.hidden_size
        self.bertConfig = bertConfig

        # token types: [Bar, Position, Pitch, Duration]
        self.n_tokens = []  # [3,18,88,66]
        self.classes = ['Bar', 'Position', 'Instrument', 'Pitch', 'Duration', 'Velocity', 'TimeSig', 'Tempo']
        for key in self.classes:
            self.n_tokens.append(len(e2w[key]))
        self.emb_sizes = [256] * 8
        self.e2w = e2w
        self.w2e = w2e

        # for deciding whether the current input_ids is a <PAD> token
        self.bar_pad_word = self.e2w['Bar']['Bar <PAD>']
        self.mask_word_np = np.array([self.e2w[etype]['%s <MASK>' % etype] for etype in self.classes], dtype=np.long)
        self.pad_word_np = np.array([self.e2w[etype]['%s <PAD>' % etype] for etype in self.classes], dtype=np.long)
        self.sos_word_np = np.array([self.e2w[etype]['%s <SOS>' % etype] for etype in self.classes], dtype=np.long)
        self.eos_word_np = np.array([self.e2w[etype]['%s <EOS>' % etype] for etype in self.classes], dtype=np.long)

        # word_emb: embeddings to change token ids into embeddings
        self.word_emb = []
        for i, key in enumerate(self.classes):  # 将每个特征都Embedding到256维，Embedding参数是可学习的
            self.word_emb.append(Embeddings(self.n_tokens[i], self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # linear layer to merge embeddings from different token types
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), bertConfig.d_model)

    def forward(self, input_ids, attn_mask=None, output_hidden_states=True):
        # convert input_ids into embeddings and merge them through linear layer
        embs = []
        for i, key in enumerate(self.classes):
            embs.append(self.word_emb[i](input_ids[..., i]))
        embs = torch.cat([*embs], dim=-1)
        emb_linear = self.in_linear(embs)

        # feed to bert
        y = self.bert(inputs_embeds=emb_linear, attention_mask=attn_mask, output_hidden_states=output_hidden_states)
        # y = y.last_hidden_state         # (batch_size, seq_len, 768)
        return y

    def get_rand_tok(self):
        rand=[0]*8
        for i in range(8):
            rand[i]=random.choice(range(self.n_tokens[i]))
        return np.array(rand)


#test
if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    config = BertConfig(max_position_embeddings=32, hidden_size=48)
    with open('../../Data/Octuple.pkl', 'rb') as f:
        e2w, w2e = pickle.load(f)
    midibert = MidiBert(config, e2w, w2e).to(device)
    input_ids_encoder = torch.randint(1, 10, (2, 32, 8)).to(device)
    encoder_attention_mask = torch.zeros((2, 32)).to(device)
    for j in range(2):
        encoder_attention_mask[j, 31] += 1
    output = midibert(input_ids_encoder, encoder_attention_mask)
    print(output.last_hidden_state.size())

