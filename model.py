import math
import numpy as np
import random
import torch
import torch.nn as nn
from transformers import BartModel,BartConfig
from PianoBart import PianoBart,Embeddings
import pickle
import torch.nn.functional as F


class PianoBartLM(nn.Module):
    def __init__(self, pianobart: PianoBart):
        super().__init__()
        self.pianobart = pianobart
        self.mask_lm = MLM(self.pianobart.e2w, self.pianobart.n_tokens, self.pianobart.hidden_size)

    def forward(self,input_ids_encoder, input_ids_decoder=None, encoder_attention_mask=None, decoder_attention_mask=None,generate=False):
        '''print(input_ids_encoder.shape)
        print(input_ids_decoder.shape)
        print(encoder_attention_mask.shape)
        print(decoder_attention_mask.shape)'''
        x = self.pianobart(input_ids_encoder, input_ids_decoder, encoder_attention_mask, decoder_attention_mask,generate)
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
        y = y.last_hidden_state
        # convert embeddings back to logits for prediction
        ys = []
        for i, etype in enumerate(self.e2w):
            ys.append(self.proj[i](y))           # (batch_size, seq_len, dict_size)
        return ys

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
        attn_mat = attn_mat.permute(0,2,1)
        return attn_mat

# class SequenceClassification(nn.Module):
#     def __init__(self, pianobart, class_num, hs, da=128, r=4):
#         super().__init__()
#         self.pianobart = pianobart
#         self.attention = SelfAttention(hs, da, r)
#         self.classifier = nn.Sequential(
#             nn.Linear(hs*r, 256),
#             nn.ReLU(),
#             nn.Linear(256, class_num)
#         )
#
#     def forward(self, input_ids_encoder, encoder_attention_mask=None):
#         x = self.pianobart(input_ids_encoder=input_ids_encoder,encoder_attention_mask=encoder_attention_mask)
#         x = x.last_hidden_state
#         attn_mat = self.attention(x)  # attn_mat: (batch, r, 512)
#         m = torch.bmm(attn_mat, x)  # m: (batch, r, 768)
#         flatten = m.view(m.size()[0], -1)  # flatten: (batch, r*768)
#         res = self.classifier(flatten)  # res: (batch, class_num)
#         return res

class SequenceClassification(nn.Module):
    def __init__(self, pianobart, class_num, hs, da=128, r=4):
        super().__init__()
        self.pianobart = pianobart
        self.attention = SelfAttention(hs, da, r)
        self.classifier = nn.Sequential(
            # nn.BatchNorm1d(hs*r),
            nn.Dropout(0.1),
            # nn.ReLU(),
            nn.Linear(hs*r, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(256, class_num)
        )

    def forward(self, input_ids_encoder, encoder_attention_mask=None):
        # y_shift = torch.zeros_like(input_ids_encoder)
        # y_shift[:, 1:, :] = input_ids_encoder[:, :-1, :]
        # y_shift[:, 0, :] = torch.tensor(self.pianobart.sos_word_np)
        # attn_shift = torch.zeros_like(encoder_attention_mask)
        # attn_shift[:, 1:] = encoder_attention_mask[:, :-1]
        # attn_shift[:, 0] = encoder_attention_mask[:, 0]
        # x = self.pianobart(input_ids_encoder=input_ids_encoder,input_ids_decoder=y_shift,encoder_attention_mask=encoder_attention_mask,decoder_attention_mask=attn_shift)

        x = self.pianobart(input_ids_encoder=input_ids_encoder,input_ids_decoder=input_ids_encoder,encoder_attention_mask=encoder_attention_mask,decoder_attention_mask=encoder_attention_mask)

        x = x.last_hidden_state
        attn_mat = self.attention(x)  # attn_mat: (batch, r, 512)
        m = torch.bmm(attn_mat, x)  # m: (batch, r, 768)
        flatten = m.view(m.size()[0], -1)  # flatten: (batch, r*768)
        res = self.classifier(flatten)  # res: (batch, class_num)
        return res


class TokenClassification(nn.Module):
    def __init__(self, pianobart, class_num, hs,d_model=64):
        super().__init__()

        self.pianobart = pianobart

        if class_num>=5: #力度预测
            new_embedding=Embeddings(n_token=class_num,d_model=d_model)
            new_linear=nn.Linear(d_model,pianobart.bartConfig.d_model)
            self.pianobart.change_decoder_embedding(new_embedding,new_linear)

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hs, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )

    def forward(self, input_ids_encoder, input_ids_decoder, encoder_attention_mask=None, decoder_attention_mask=None):
        x = self.pianobart(input_ids_encoder, input_ids_decoder, encoder_attention_mask, decoder_attention_mask)
        x = x.last_hidden_state
        res = self.classifier(x)
        return res

#test
if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    config=BartConfig(max_position_embeddings=32, d_model=48)
    with open('./Data/Octuple.pkl', 'rb') as f:
        e2w, w2e = pickle.load(f)
    piano_bart=PianoBart(config,e2w,w2e).to(device)
    input_ids_encoder = torch.randint(0, 10, (2, 32, 8)).to(device)
    print("输入维度:",input_ids_encoder.size())
    input_ids_decoder = torch.randint(0, 10, (2, 32, 8)).to(device)
    # label = torch.randint(0, 10, (2, 32)).to(device)
    label = torch.randint(0, 10, (2, 32, 8)).to(device)
    encoder_attention_mask = torch.zeros((2, 32)).to(device)
    decoder_attention_mask = torch.zeros((2, 32)).to(device)
    for j in range(2):
        encoder_attention_mask[j, 31] += 1
        decoder_attention_mask[j, 31] += 1
        decoder_attention_mask[j, 30] += 1

    test_PianoBart=False
    if test_PianoBart:
        print("test PianoBart")
        piano_bart_lm=PianoBartLM(piano_bart).to(device)
        #print(piano_bart_lm)
        output=piano_bart_lm(input_ids_encoder,input_ids_decoder,encoder_attention_mask,decoder_attention_mask)
        print("输出维度:")
        for temp in output:
            print(temp.size())

    test_TokenClassifier=False
    if test_TokenClassifier:
        print("test Token Classifier")
        piano_bart_token_classifier=TokenClassification(pianobart=piano_bart, class_num=10, hs=48)
        output=piano_bart_token_classifier(input_ids_encoder,label,encoder_attention_mask,decoder_attention_mask)
        print("输出维度:",output.size())

    test_SequenceClassifier=True
    if test_SequenceClassifier:
        print("test Sequence Classifier")
        piano_bart_sequence_classifier=SequenceClassification(pianobart=piano_bart, class_num=10, hs=48)
        output=piano_bart_sequence_classifier(input_ids_encoder,encoder_attention_mask)
        print("输出维度:",output.size())