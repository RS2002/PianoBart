import math
import numpy as np
import random

import torch
import torch.nn as nn
from transformers import BertModel

from MidiBert import MidiBert
import pickle
from transformers import BertModel,BertConfig
import torch.nn.functional as F




class MidiBertLM(nn.Module):
    def __init__(self, midibert: MidiBert):
        super().__init__()

        self.midibert = midibert
        self.mask_lm = MLM(self.midibert.e2w, self.midibert.n_tokens, self.midibert.hidden_size)

    def forward(self, x, attn):
        x = self.midibert(x, attn)
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
        # feed to bert
        y = y.hidden_states[-1]

        # convert embeddings back to logits for prediction
        ys = []
        for i, etype in enumerate(self.e2w):
            ys.append(self.proj[i](y))  # (batch_size, seq_len, dict_size)
        return ys

class TokenClassification(nn.Module):
    def __init__(self, midibert, class_num, hs):
        super().__init__()

        self.midibert = midibert
        self.classifier = nn.Sequential(
            #nn.Dropout(0.1),
            nn.Linear(hs, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )

    def forward(self, y, attn, layer=-1):
        # feed to bert
        y = self.midibert(y, attn, output_hidden_states=True)
        # y = y.last_hidden_state         # (batch_size, seq_len, 768)
        y = y.hidden_states[layer]
        return self.classifier(y)

class SequenceClassification(nn.Module):
    def __init__(self, midibert, class_num, hs, da=128, r=4):
        super(SequenceClassification, self).__init__()
        self.midibert = midibert
        self.attention = SelfAttention(hs, da, r)
        self.classifier = nn.Sequential(
            #nn.Linear(hs * r, 256),
            nn.Linear(hs, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )

    def forward(self, x, attn, layer=-1):  # x: (batch, 512, 4)
        x = self.midibert(x, attn, output_hidden_states=True)  # (batch, 512, 768)
        # y = y.last_hidden_state         # (batch_size, seq_len, 768)
        '''x = x.hidden_states[layer]
        attn_mat = self.attention(x)  # attn_mat: (batch, r, 512)
        m = torch.bmm(attn_mat, x)  # m: (batch, r, 768)
        flatten = m.view(m.size()[0], -1)  # flatten: (batch, r*768)
        res = self.classifier(flatten)  # res: (batch, class_num)
        return res'''
        return self.classifier(x.hidden_states[layer][:,0,:])

class SelfAttention(nn.Module):
    def __init__(self, input_dim, da, r):
        '''
        Args:
            input_dim (int): batch, seq, input_dim
            da (int): number of features in hidden layer from self-attn
            r (int): number of aspects of self-attn
        '''
        super(SelfAttention, self).__init__()
        self.ws1 = nn.Linear(input_dim, da, bias=False)
        self.ws2 = nn.Linear(da, r, bias=False)

    def forward(self, h):
        attn_mat = F.softmax(self.ws2(torch.tanh(self.ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0, 2, 1)
        return attn_mat


#test
if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    config = BertConfig(max_position_embeddings=32, hidden_size=48)
    with open('../../Data/Octuple.pkl', 'rb') as f:
        e2w, w2e = pickle.load(f)
    midibert = MidiBert(config, e2w, w2e).to(device)
    input_ids_encoder = torch.randint(1, 10, (2, 32, 8)).to(device)
    print("输入维度:", input_ids_encoder.size())
    encoder_attention_mask = torch.zeros((2, 32)).to(device)
    for j in range(2):
        encoder_attention_mask[j, 31] += 1
    label = torch.randint(1, 10, (2, 32)).to(device)

    test_MidiBert=False
    if test_MidiBert:
        print("test PianoBart")
        piano_bart_lm=MidiBertLM(midibert).to(device)
        #print(piano_bart_lm)
        output=piano_bart_lm(input_ids_encoder,encoder_attention_mask)
        print("输出维度:")
        for temp in output:
            print(temp.size())

    test_TokenClassifier=False
    if test_TokenClassifier:
        print("test Token Classifier")
        piano_bart_token_classifier=TokenClassification(midibert=midibert, class_num=10, hs=48)
        output=piano_bart_token_classifier(input_ids_encoder,encoder_attention_mask)
        print("输出维度:",output.size())

    test_SequenceClassifier=False
    if test_SequenceClassifier:
        print("test Sequence Classifier")
        piano_bart_sequence_classifier=SequenceClassification(midibert=midibert, class_num=10, hs=48)
        output=piano_bart_sequence_classifier(input_ids_encoder,encoder_attention_mask)
        print("输出维度:",output.size())


