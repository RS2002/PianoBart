import math
import numpy as np
import random
import torch
import torch.nn as nn
from transformers import BartModel,BartConfig
import pickle

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super().__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PianoBart(nn.Module):
    def __init__(self, bartConfig, e2w, w2e):
        super().__init__()

        self.bart = BartModel(bartConfig)
        self.hidden_size = bartConfig.d_model # Dimensionality of the layers and the pooler layer
        self.bartConfig = bartConfig

        # token types: 0 Measure（第几个Bar（小节））, 1 Position（Bar中的位置）, 2 Program（乐器）, 3 Pitch（音高）, 4 Duration（持续时间）, 5 Velocity（力度）, 6 TimeSig（拍号）, 7 Tempo（速度）
        self.n_tokens = []  # 每个属性的种类数
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
            self.word_emb.append(Embeddings(512, self.emb_sizes[i]))
        self.word_emb = nn.ModuleList(self.word_emb)

        # linear layer to merge embeddings from different token types
        self.encoder_linear = nn.Linear(np.sum(self.emb_sizes), bartConfig.d_model)
        self.decoder_linear = self.encoder_linear
        #self.decoder_linear= nn.Linear(np.sum(self.emb_sizes), bartConfig.d_model)

    def forward(self, input_ids_encoder, input_ids_decoder, encoder_attention_mask=None, decoder_attention_mask=None, output_hidden_states=True):
        # convert input_ids into embeddings and merge them through linear layer
        encoder_embs = []
        decoder_embs = []
        for i, key in enumerate(self.classes):
            print(self.word_emb[i])
            encoder_embs.append(self.word_emb[i](input_ids_encoder[..., i]))
            decoder_embs.append(self.word_emb[i](input_ids_decoder[..., i]))
        encoder_embs = torch.cat([*encoder_embs], dim=-1)
        emb_linear_encoder = self.encoder_linear(encoder_embs)
        decoder_embs = torch.cat([*decoder_embs], dim=-1)
        emb_linear_decoder = self.decoder_linear(decoder_embs)
        print('emb_lin', emb_linear_encoder.shape)
        print('emb_lin_dec', emb_linear_decoder.shape)
        # feed to bart
        y = self.bart(inputs_embeds=emb_linear_encoder, decoder_inputs_embeds=emb_linear_decoder, attention_mask=encoder_attention_mask, decoder_attention_mask=decoder_attention_mask, output_hidden_states=output_hidden_states) #attention_mask用于屏蔽<PAD> (PAD作用是在结尾补齐长度)
        # y = y.last_hidden_state         # (batch_size, seq_len, 1536)
        return y

    def get_rand_tok(self):
        rand=[0]*8
        for i in range(8):
            rand[i]=random.choice(range(self.n_tokens[i]))
        return np.array(rand)

#test
if __name__=='__main__':
    device = torch.device("cuda")
    config=BartConfig(max_position_embeddings=32, d_model=48)
    with open('./Data/Octuple.pkl', 'rb') as f:
        e2w, w2e = pickle.load(f)
    piano_bart=PianoBart(config,e2w,w2e).to(device)
    #print(piano_bart)
    input_ids_encoder = torch.randint(1, 10, (2, 32, 8)).to(device)
    input_ids_decoder = torch.randint(1, 10, (2, 32, 8)).to(device)
    encoder_attention_mask = torch.zeros((2, 32)).to(device)
    decoder_attention_mask = torch.zeros((2, 32)).to(device)
    for j in range(2):
        encoder_attention_mask[j, 31] += 1
        decoder_attention_mask[j, 31] += 1
        decoder_attention_mask[j, 30] += 1
    output=piano_bart(input_ids_encoder,input_ids_decoder,encoder_attention_mask,decoder_attention_mask)
    print(output.last_hidden_state.size())
