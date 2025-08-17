import numpy as np
import torch
import torch.nn as nn
from PianoBart import PianoBart
import torch.nn.functional as F

class PianoBartLM(nn.Module):
    def __init__(self, pianobart: PianoBart):
        super().__init__()
        self.pianobart = pianobart
        self.mask_lm = MLM(self.pianobart.e2w, self.pianobart.n_tokens, self.pianobart.hidden_size)

    def forward(self,input_ids_encoder, input_ids_decoder=None, encoder_attention_mask=None, decoder_attention_mask=None,generate=False,device_num=-1):
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
    cur_word = nucleus(probs, p=p)
    return cur_word


class MLM(nn.Module):
    def __init__(self, e2w, n_tokens, hidden_size):
        super().__init__()
        self.proj = []
        for i, etype in enumerate(e2w):
            self.proj.append(nn.Linear(hidden_size, n_tokens[i]))
        self.proj = nn.ModuleList(self.proj)
        self.e2w = e2w

    def forward(self, y):
        y = y.last_hidden_state
        ys = []
        for i, etype in enumerate(self.e2w):
            ys.append(self.proj[i](y))
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


class SequenceClassification(nn.Module):
    def __init__(self, pianobart, class_num, hs, da=128, r=4):
        super().__init__()
        self.pianobart = pianobart
        self.attention = SelfAttention(hs, da, r)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hs*r, 256),
            nn.ReLU(),
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
        attn_mat = self.attention(x)
        m = torch.bmm(attn_mat, x)
        flatten = m.view(m.size()[0], -1)
        res = self.classifier(flatten)
        return res


class TokenClassification(nn.Module):
    def __init__(self, pianobart, class_num, hs):
        super().__init__()
        self.pianobart = pianobart
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
