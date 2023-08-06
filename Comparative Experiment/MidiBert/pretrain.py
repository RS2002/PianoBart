import torch
import torch.nn as nn
from transformers import AdamW
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random
import tqdm
import sys
import shutil
import copy
from MidiBert import MidiBert
from model import MidiBertLM
from transformers import BartConfig
import pickle
import argparse
import os


def get_args_pretrain():
    parser = argparse.ArgumentParser(description='')

    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='../../Data/Octuple.pkl')
    parser.add_argument('--name', type=str, default='midibert')

    ### pre-train dataset ###
    parser.add_argument("--datasets", type=str, nargs='+', default=['asap', 'EMOPIA', 'Pianist8', 'POP1K7', 'POP909'])

    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--mask_percent', type=float, default=0.15,
                        help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction")
    parser.add_argument('--max_seq_len', type=int, default=1024, help='all sequences are padded to `max_seq_len`') #TODO:512
    parser.add_argument('--hs', type=int, default=768)  # hidden state
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')

    ### cuda ###
    parser.add_argument("--cpu", action="store_true")  # default: False
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0,1], help="CUDA device ids")

    args = parser.parse_args()

    return args



class Pretrainer:
    def __init__(self, midibert: MidiBert, train_dataloader, valid_dataloader,
                 lr, batch, max_seq_len, mask_percent, cpu, cuda_devices=None):
        device_name="cuda"
        if cuda_devices is not None and len(cuda_devices)>=1:
            device_name+=":"+str(cuda_devices[0])
        self.device = torch.device(device_name if torch.cuda.is_available() and not cpu else 'cpu')
        self.midibert = midibert.to(self.device)  # save this for ckpt
        self.model = MidiBertLM(midibert).to(self.device)
        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('# total parameters:', self.total_params)

        if len(cuda_devices) > 1 and not cpu:
            print("Use %d GPUS" % len(cuda_devices) )
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        elif (len(cuda_devices)  == 1 or torch.cuda.is_available()) and not cpu:
            print("Use GPU" , end=" ")
        else:
            print("Use CPU")

        print(self.device)


        self.train_data = train_dataloader
        self.valid_data = valid_dataloader

        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.batch = batch
        self.max_seq_len = max_seq_len
        self.mask_percent = mask_percent
        self.Lseq = [i for i in range(self.max_seq_len)]
        self.Lseq_element = [i for i in range(self.max_seq_len * 8)]
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def train(self):
        self.model.train()
        train_loss, train_acc = self.iteration(self.train_data, self.max_seq_len)
        return train_loss, train_acc

    def valid(self):
        self.model.eval()
        valid_loss, valid_acc = self.iteration(self.valid_data, self.max_seq_len, train=False)
        return valid_loss, valid_acc

    def save_checkpoint(self, epoch, best_acc, valid_acc,
                        valid_loss, train_loss, is_best, filename):
        state = {
            'epoch': epoch + 1,
            'state_dict': self.midibert.state_dict(),
            'best_acc': best_acc,
            'valid_acc': valid_acc,
            'valid_loss': valid_loss,
            'train_loss': train_loss,
            'optimizer': self.optim.state_dict()
        }
        torch.save(state, filename)
        best_mdl = filename.split('.')[0] + '_best.ckpt'
        if is_best:
            shutil.copyfile(filename, best_mdl)

    def compute_loss(self, predict, target, loss_mask):
        '''print(predict.type)
        print(target.type)'''
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        #TODO: add weights for different attributes
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def get_mask_ind(self):
        mask_ind = random.sample(self.Lseq, round(self.max_seq_len * self.mask_percent))
        mask80 = random.sample(mask_ind, round(len(mask_ind)*0.8))
        left = list(set(mask_ind)-set(mask80))
        rand10 = random.sample(left, round(len(mask_ind)*0.1))
        cur10 = list(set(left)-set(rand10))
        return mask80, rand10, cur10

    def iteration(self, training_data, max_seq_len, train=True):
        pbar = tqdm.tqdm(training_data, disable=False)  # 显示进度条

        total_acc, total_losses = [0] * len(self.midibert.e2w), 0

        for ori_seq_batch in pbar:
            batch = ori_seq_batch.shape[0]
            ori_seq_batch = ori_seq_batch.type(torch.LongTensor).to(self.device)
            input_ids = copy.deepcopy(ori_seq_batch)
            loss_mask = torch.zeros(batch, max_seq_len)

            for b in range(batch):
                # get index for masking
                mask80, rand10, cur10 = self.get_mask_ind()
                # apply mask, random, remain current token
                for i in mask80:
                    mask_word = torch.tensor(self.midibert.mask_word_np).to(self.device)
                    input_ids[b][i] = mask_word
                    loss_mask[b][i] = 1
                for i in rand10:
                    rand_word = torch.tensor(self.midibert.get_rand_tok()).to(self.device)
                    input_ids[b][i] = rand_word
                    loss_mask[b][i] = 1
                for i in cur10:
                    loss_mask[b][i] = 1

            loss_mask = loss_mask.to(self.device)

            # avoid attend to pad word
            attn_mask = (input_ids[:, :, 0] != self.midibert.bar_pad_word).float().to(self.device)  # (batch, seq_len)
            input_ids[:, :, 2] = 0
            input_ids[:, :, 5] = 0
            input_ids[:, :, 6] = 0
            input_ids[:, :, 7] = 0
            y = self.model.forward(input_ids, attn_mask)

            # get the most likely choice with max
            outputs = []
            for i, etype in enumerate(self.midibert.e2w):
                output = np.argmax(y[i].cpu().detach().numpy(), axis=-1)
                outputs.append(output)
            outputs = np.stack(outputs, axis=-1)
            outputs = torch.from_numpy(outputs).to(self.device)  # (batch, seq_len)

            # accuracy 只考虑mask部分输出的准确率
            all_acc = []
            for i in [0,1,3,4]:
                acc = torch.sum((ori_seq_batch[:, :, i] == outputs[:, :, i]).float() * loss_mask)
                acc /= torch.sum(loss_mask)
                all_acc.append(acc)
            total_acc = [sum(x) for x in zip(total_acc, all_acc)]

            # reshape (b, s, f) -> (b, f, s)
            for i, etype in enumerate(self.midibert.e2w):
                # print('before',y[i][:,...].shape)   # each: (4,512,5), (4,512,20), (4,512,90), (4,512,68)
                y[i] = y[i][:, ...].permute(0, 2, 1)  # 维度交换

            # calculate losses
            losses, n_tok = [], []
            for i, etype in enumerate(self.midibert.e2w):
                if i in [0,1,3,4]:
                    n_tok.append(len(self.midibert.e2w[etype]))
                    losses.append(self.compute_loss(y[i], ori_seq_batch[..., i], loss_mask))
            total_loss_all = [x * y for x, y in zip(losses, n_tok)]
            total_loss = sum(total_loss_all) / sum(n_tok)  # weighted

            # udpate only in train
            if train:
                self.model.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.model.parameters(), 3.0)  # 用于裁剪梯度，防止梯度爆炸
                self.optim.step()

            # acc
            accs = list(map(float, all_acc))
            sys.stdout.write(
                'Loss: {:06f} | loss: {:03f}, {:03f}, {:03f}, {:03f} | acc: {:03f}, {:03f}, {:03f}, {:03f} \n'.format(
                    total_loss, *losses, *accs))
            #print( total_loss, losses, accs)

            losses = list(map(float, losses))
            total_losses += total_loss.item()

        return round(total_losses / len(training_data), 3), [round(x.item() / len(training_data), 3) for x in total_acc]

def load_data_pretrain(datasets,mode):
    if mode=="pretrain":
        to_concat = []
        root = '../../Data/output'

        # for dataset in datasets:
        #     data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)
        #     print(f'   {dataset}: {data.shape}')
        #     to_concat.append(data)
        for dataset in datasets:
            data_train = np.load(os.path.join(root, dataset, 'midi_train_split.npy'), allow_pickle = True)
            data_test = np.load(os.path.join(root, dataset, 'midi_test_split.npy'), allow_pickle = True)
            data_valid = np.load(os.path.join(root, dataset, 'midi_valid_split.npy'), allow_pickle = True)
            data = np.concatenate((data_train, data_test, data_valid), axis = 0)
            print(f'   {dataset}: {data.shape}')
            to_concat.append(data)

        training_data = np.vstack(to_concat)
        print('   > all training data:', training_data.shape)
        # shuffle during training phase
        index = np.arange(len(training_data))
        np.random.shuffle(index)
        training_data = training_data[index]
        split = int(len(training_data) * 0.85)
        X_train, X_val = training_data[:split], training_data[split:]
        return X_train, X_val

    else:
        return None