import copy
import math
import numpy as np
import random
import torch
import torch.nn as nn
from transformers import BartModel,BartConfig
from PianoBart import PianoBart,Embeddings
import pickle
import torch.nn.functional as F
import tqdm


class PianoBartLM(nn.Module):
    def __init__(self, pianobart: PianoBart):
        super().__init__()
        self.pianobart = pianobart
        self.mask_lm = MLM(self.pianobart.e2w, self.pianobart.n_tokens, self.pianobart.hidden_size)

    def forward(self,input_ids_encoder, input_ids_decoder=None, encoder_attention_mask=None, decoder_attention_mask=None,generate=False,device_num=-1):
        '''print(input_ids_encoder.shape)
        print(input_ids_decoder.shape)
        print(encoder_attention_mask.shape)
        print(decoder_attention_mask.shape)'''
        if not generate:
            x = self.pianobart(input_ids_encoder, input_ids_decoder, encoder_attention_mask, decoder_attention_mask)
            return self.mask_lm(x)
        else:
            if input_ids_encoder.shape[0] !=1:
                print("ERROR")
                exit(-1)
            if device_num==-1:
                device=torch.device('cpu')
            else:
                device=torch.device('cuda:'+str(device_num))
            pad=torch.from_numpy(self.pianobart.pad_word_np)
            input_ids_decoder=pad.repeat(input_ids_encoder.shape[0],input_ids_encoder.shape[1],1).to(device)
            result=pad.repeat(input_ids_encoder.shape[0],input_ids_encoder.shape[1],1).to(device)
            decoder_attention_mask=torch.zeros_like(encoder_attention_mask).to(device)
            input_ids_decoder[:,0,:] = torch.tensor(self.pianobart.sos_word_np)
            decoder_attention_mask[:,0] = 1
            for i in range(input_ids_encoder.shape[1]):
            # pbar = tqdm.tqdm(range(input_ids_encoder.shape[1]), disable=False)
            # for i in pbar:
                x = self.mask_lm(self.pianobart(input_ids_encoder, input_ids_decoder, encoder_attention_mask, decoder_attention_mask))
                # outputs = []
                # for j, etype in enumerate(self.pianobart.e2w):
                #     output = np.argmax(x[j].cpu().detach().numpy(), axis=-1)
                #     outputs.append(output)
                # outputs = np.stack(outputs, axis=-1)
                # outputs = torch.from_numpy(outputs)
                # outputs=self.sample(x)
                # if i!=input_ids_encoder.shape[1]-1:
                #     input_ids_decoder[:,i+1,:]=outputs[:,i,:]
                #     decoder_attention_mask[:,i+1]+=1
                # result[:,i,:]=outputs[:,i,:]
                current_output=self.sample(x,i)
                # print(current_output)
                if i!=input_ids_encoder.shape[1]-1:
                    input_ids_decoder[:,i+1,:]=current_output
                    decoder_attention_mask[:,i+1]+=1
                # 为提升速度，提前终止生成
                if (current_output>=pad).any():
                    break
                result[:,i,:]=current_output
            return result

    def sample(self,x,index): # Adaptive Sampling Policy in CP Transformer
        # token types: 0 Measure（第几个Bar（小节））, 1 Position（Bar中的位置）, 2 Program（乐器）, 3 Pitch（音高）, 4 Duration（持续时间）, 5 Velocity（力度）, 6 TimeSig（拍号）, 7 Tempo（速度）
        t=[1.2,1.2,5,1,2,5,5,1.2]
        p=[1,1,1,0.9,0.9,1,1,0.9]
        result=[]
        for j, etype in enumerate(self.pianobart.e2w):
            y=x[j]
            y=y[:,index,:]
            y=sampling(y,p[j],t[j])
            result.append(y)
        return torch.tensor(result)




# -- nucleus -- #
def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[0:1]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


def sampling(logit, p=None, t=1.0):
    logit = logit.squeeze()
    probs = torch.softmax(logit/t,dim=-1)
    probs=probs.cpu().detach().numpy()
    #print(probs.shape)
    cur_word = nucleus(probs, p=p)
    return cur_word

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
            # Excitation(hs*r),
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

        '''self.attention = SelfAttention(hs*2, da, r)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            # Excitation(hs*r*2),
            nn.Linear(hs*r*2, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )'''

    def forward(self, input_ids_encoder, encoder_attention_mask=None):
        # y_shift = torch.zeros_like(input_ids_encoder)
        # y_shift[:, 1:, :] = input_ids_encoder[:, :-1, :]
        # y_shift[:, 0, :] = torch.tensor(self.pianobart.sos_word_np)
        # attn_shift = torch.zeros_like(encoder_attention_mask)
        # attn_shift[:, 1:] = encoder_attention_mask[:, :-1]
        # attn_shift[:, 0] = encoder_attention_mask[:, 0]
        # x = self.pianobart(input_ids_encoder=input_ids_encoder,input_ids_decoder=y_shift,encoder_attention_mask=encoder_attention_mask,decoder_attention_mask=attn_shift)

        x = self.pianobart(input_ids_encoder=input_ids_encoder,input_ids_decoder=input_ids_encoder,encoder_attention_mask=encoder_attention_mask,decoder_attention_mask=encoder_attention_mask)
        # x=self.pianobart(input_ids_encoder=input_ids_encoder,encoder_attention_mask=encoder_attention_mask)

        x = x.last_hidden_state

        # x=x.encoder_last_hidden_state

        # x = torch.cat([x.last_hidden_state, x.encoder_last_hidden_state], dim=-1)


        attn_mat = self.attention(x)  # attn_mat: (batch, r, 512)
        m = torch.bmm(attn_mat, x)  # m: (batch, r, 768)
        flatten = m.view(m.size()[0], -1)  # flatten: (batch, r*768)
        res = self.classifier(flatten)  # res: (batch, class_num)
        return res

class Excitation(nn.Module):
    def __init__(self,channel_dim,reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel_dim, channel_dim // reduction),
            nn.ReLU(),
            nn.Linear(channel_dim // reduction, channel_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y # + x



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
            # Excitation(hs),
            nn.Linear(hs, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )

        '''self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            # Excitation(hs*2),
            nn.Linear(hs*2, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )'''

    def forward(self, input_ids_encoder, input_ids_decoder, encoder_attention_mask=None, decoder_attention_mask=None):
        x = self.pianobart(input_ids_encoder, input_ids_decoder, encoder_attention_mask, decoder_attention_mask)
        x = x.last_hidden_state

        # x=x.encoder_last_hidden_state

        # x = torch.cat([x.last_hidden_state, x.encoder_last_hidden_state], dim=-1)

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
    label = torch.randint(0, 10, (2, 32)).to(device)
    encoder_attention_mask = torch.zeros((2, 32)).to(device)
    decoder_attention_mask = torch.zeros((2, 32)).to(device)
    for j in range(2):
        encoder_attention_mask[j, 31] += 1
        decoder_attention_mask[j, 31] += 1
        decoder_attention_mask[j, 30] += 1

    test_PianoBart=False
    if test_PianoBart:
        print("test PianoBART")
        piano_bart_lm=PianoBartLM(piano_bart).to(device)
        output=piano_bart_lm(input_ids_encoder,input_ids_decoder,encoder_attention_mask,decoder_attention_mask)
        print("输出维度:")
        for temp in output:
            print(temp.size())

    test_generate=True
    if test_generate:
        print("test generation")
        piano_bart_lm=PianoBartLM(piano_bart).to(device)
        output=piano_bart_lm(input_ids_encoder = input_ids_encoder, encoder_attention_mask = encoder_attention_mask, generate = True)
        print("输出维度:")
        print(output.shape)

    test_TokenClassifier=False
    if test_TokenClassifier:
        print("test Token Classifier")
        piano_bart_token_classifier=TokenClassification(pianobart=piano_bart, class_num=10, hs=48)
        output=piano_bart_token_classifier(input_ids_encoder,label,encoder_attention_mask,decoder_attention_mask)
        print("输出维度:",output.size())

    test_SequenceClassifier=False
    if test_SequenceClassifier:
        print("test Sequence Classifier")
        piano_bart_sequence_classifier=SequenceClassification(pianobart=piano_bart, class_num=10, hs=48)
        output=piano_bart_sequence_classifier(input_ids_encoder,encoder_attention_mask)
        print("输出维度:",output.size())