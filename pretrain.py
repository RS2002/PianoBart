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
from PianoBart import PianoBart
from model import PianoBartLM
from transformers import BartConfig
import pickle
import argparse
import os

def get_args_pretrain():
    parser = argparse.ArgumentParser(description='')

    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='./Data/Octuple.pkl')
    parser.add_argument('--name', type=str, default='PianoBart')

    ### pre-train dataset ###
    parser.add_argument("--datasets", type=str, nargs='+', default=['asap', 'EMOPIA', 'Pianist8', 'POP1K7', 'POP909']) #TODO

    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--mask_percent', type=float, default=0.15,
                        help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction")
    parser.add_argument('--max_seq_len', type=int, default=1024, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=1024)  # hidden state
    parser.add_argument('--layers', type=int, default=8)  # layer nums of encoder & decoder
    parser.add_argument('--ffn_dims', type=int, default=2048)  # FFN dims
    parser.add_argument('--heads', type=int, default=8)  # attention heads

    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')

    ### cuda ###
    parser.add_argument("--cpu", action="store_true")  # default: False
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0,1], help="CUDA device ids")

    args = parser.parse_args()

    return args


class Pretrainer:
    def __init__(self, pianobart: PianoBart, train_dataloader, valid_dataloader,
                 lr, batch, max_seq_len, mask_percent, cpu, cuda_devices=None):
        device_name="cuda"
        if cuda_devices is not None and len(cuda_devices)>=1:
            device_name+=":"+str(cuda_devices[0])
        self.device = torch.device(device_name if torch.cuda.is_available() and not cpu else 'cpu')
        self.pianobart = pianobart.to(self.device)  # save this for ckpt
        self.model = PianoBartLM(pianobart).to(self.device)
        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('# total parameters:', self.total_params)

        if len(cuda_devices) > 1 and not cpu:
            print("Use %d GPUS" % len(cuda_devices) )
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        elif (len(cuda_devices)  == 1 or torch.cuda.is_available()) and not cpu:
            print("Use GPU" , end=" ")
            print(self.device)
        else:
            print("Use CPU")


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
            'state_dict': self.pianobart.state_dict(),
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

    def iteration(self, training_data, max_seq_len, train=True):
        pbar = tqdm.tqdm(training_data, disable=False)  # 显示进度条

        total_acc, total_losses = [0] * len(self.pianobart.e2w), 0

        for ori_seq_batch in pbar:
            batch = ori_seq_batch.shape[0]
            ori_seq_batch = ori_seq_batch.type(torch.LongTensor).to(self.device)  # (batch, seq_len, 8)
            input_ids_encoder = copy.deepcopy(ori_seq_batch)
            input_ids_decoder = torch.zeros_like(input_ids_encoder)
            loss_mask = torch.zeros(batch, max_seq_len, 8)
            for b in range(batch):
                shifted_input_ids = input_ids_encoder[b].new_zeros(input_ids_encoder[b].shape)
                # print(input_ids_encoder.shape)
                # print(shifted_input_ids.shape)
                # print(self.pianobart.sos_word_np.shape)
                # print(input_ids_encoder[b][:, :-1].shape)
                shifted_input_ids[1:] = input_ids_encoder[b][:-1, :].clone()
                shifted_input_ids[0] = torch.tensor(self.pianobart.sos_word_np)
                input_ids_decoder[b] = shifted_input_ids
                input_mask, mask_pos = self.gen_mask(input_ids_encoder[b].cpu())
                if mask_pos.size()[-1] != 8:
                    mask_pos = np.repeat(mask_pos[:, np.newaxis], 8, axis=1)
                input_ids_encoder[b] = input_mask
                loss_mask[b] = mask_pos

            input_ids_encoder = input_ids_encoder.to(self.device)
            input_ids_decoder = input_ids_decoder.to(self.device)
            loss_mask = loss_mask.to(self.device)
            # avoid attend to pad word
            #print(input_ids_encoder.shape)
            encoder_attention_mask = (input_ids_encoder[:, :, 0] != self.pianobart.bar_pad_word).float().to(
                self.device)  # (batch, seq_len)
            decoder_attention_mask = (input_ids_decoder[:, :, 0] != self.pianobart.bar_pad_word).float().to(self.device)

            '''print(input_ids_encoder.shape)
            print(input_ids_decoder.shape)
            print(encoder_attention_mask.shape)
            print(decoder_attention_mask.shape)
            print(input_ids_encoder.device)
            print(input_ids_decoder.device)
            print(encoder_attention_mask.device)
            print(encoder_attention_mask.device)'''
            # tmp_tensor = torch.zeros_like(input_ids_encoder.shape).to(self.device)
            # tmp_tensor1 = torch.zeros_like(input_ids_decoder.shape).to(self.device)
            # tmp_tensor2 = torch.zeros_like(encoder_attention_mask.shape).to(self.device)
            # tmp_tensor3 = torch.zeros_like(encoder_attention_mask.shape).to(self.device)

            # y = []
            # for ids in range(input_ids_encoder.shape[0]):
            #     y_t = self.model.forward(input_ids_encoder[ids], input_ids_decoder[ids], encoder_attention_mask, decoder_attention_mask)
            #     y.append(y_t)
            y = self.model.forward(input_ids_encoder, input_ids_decoder, encoder_attention_mask, decoder_attention_mask)
            # y = torch.Tensor(y).to(self.device)

            # get the most likely choice with max
            outputs = []
            for i, etype in enumerate(self.pianobart.e2w):
                output = np.argmax(y[i].cpu().detach().numpy(), axis=-1)
                outputs.append(output)
            outputs = np.stack(outputs, axis=-1)
            outputs = torch.from_numpy(outputs).to(self.device)  # (batch, seq_len)

            # accuracy 只考虑mask部分输出的准确率
            all_acc = []
            for i in range(8):
                acc = torch.sum((ori_seq_batch[:, :, i] == outputs[:, :, i]).float() * loss_mask[:, :, i])
                acc /= torch.sum(loss_mask[:, :, i])
                all_acc.append(acc)
            total_acc = [sum(x) for x in zip(total_acc, all_acc)]

            # reshape (b, s, f) -> (b, f, s)
            for i, etype in enumerate(self.pianobart.e2w):
                # print('before',y[i][:,...].shape)   # each: (4,512,5), (4,512,20), (4,512,90), (4,512,68)
                y[i] = y[i][:, ...].permute(0, 2, 1)  # 维度交换

            # calculate losses
            losses, n_tok = [], []
            for i, etype in enumerate(self.pianobart.e2w):
                n_tok.append(len(self.pianobart.e2w[etype]))
                losses.append(self.compute_loss(y[i], ori_seq_batch[..., i], loss_mask[:, :, i]))
            total_loss_all = [x * y for x, y in zip(losses, n_tok)]
            total_loss = sum(total_loss_all) / sum(n_tok)  # weighted

            # update only in train
            if train:
                self.model.zero_grad()
                total_loss.backward()
                clip_grad_norm_(self.model.parameters(), 3.0)  # 用于裁剪梯度，防止梯度爆炸
                self.optim.step()

            # acc
            accs = list(map(float, all_acc))
            sys.stdout.write(
                'Loss: {:06f} | loss: {:03f}, {:03f}, {:03f}, {:03f}, {:03f}, {:03f}, {:03f}, {:03f}\n'.format(
                    total_loss, *losses))
            sys.stdout.write(
                'Acc: {:06f} | acc: {:03f}, {:03f}, {:03f}, {:03f}, {:03f}, {:03f}, {:03f}, {:03f}\n'.format(np.average(accs),*accs))

            losses = list(map(float, losses))
            total_losses += total_loss.item()

        return round(total_losses / len(training_data), 3), [round(x.item() / len(training_data), 3) for x in total_acc]

    def gen_mask(self, input_ids: torch.Tensor, choice=None):
        # for now, n is only used to represent to delete or masked
        # if n == -1, delete
        # else masked
        # TODO
        # more detailed mask, like mask 1/8, 1/4, 1/2, 1/1 (1/n)
        def TokenDeletion(input_ids: torch.Tensor, mask_percent, replacement: np.array, n=-1):
            def deleteOctuple(input_ids: torch.Tensor, mask_percent, replacement: np.array):
                l = input_ids.shape[0]
                length = int(l * mask_percent)
                maskpos = [1 if i < length else 0 for i in range(l)]
                random.shuffle(maskpos)
                maskpos = np.array(maskpos)
                masked = copy.deepcopy(input_ids).numpy()
                count = 0
                for i in range(len(maskpos)):
                    if maskpos[i] == 1:
                        masked = np.delete(masked, i - count, axis=0)
                        count += 1
                pos = np.where(maskpos == 1)[0]
                if len(pos) > 0:
                    pos = pos[0]
                    maskpos[pos:] = 1
                for i in range(length):
                    masked = np.append(masked, replacement.reshape(1, 8), axis=0)
                return torch.from_numpy(masked), torch.from_numpy(maskpos)

            if n == -1:
                return deleteOctuple(input_ids, mask_percent, replacement)
            else:
                # TODO
                # TEST CASE REQUIRED
                # OCTUPLE
                # (Bar, Pos, Program, Pitch, Duration, Velocity, TimeSignature, Tempo)
                masked = copy.deepcopy(input_ids).numpy()
                barMax = masked[-1, 0]
                length = int(barMax * mask_percent)
                maskBarPos = [1 if i < length else 0 for i in range(barMax)]
                random.shuffle(maskBarPos)
                maskBarPos = np.array(maskBarPos)
                count = 0
                for i in range(len(masked)):
                    if maskBarPos[masked[i, 0]] == 1:
                        masked = np.delete(masked, i - count, axis=0)
                        count += 1
                for i in range(length):
                    masked = np.append(masked, replacement.reshape(1, 8), axis=0)
                return torch.from_numpy(masked), torch.from_numpy(maskBarPos)

        # for now, n is only used to represent to Octuple-level mask or Bar-level mask
        # if n == 0, Octuple-level mask
        # else Bar-level mask
        # if element_level=True, element-level mask
        # else Octuple-level mask
        # TODO
        # more detailed mask, like mask 1/8, 1/4, 1/2, 1/1 (1/n) bar
        def TokenMask(input_ids: torch.Tensor, mask_percent, n=-1, element_level=False):
            def generate_mask(sz, prob):
                mask_n = np.random.rand(sz)
                mask_s = np.zeros(sz, dtype=np.int8)
                mask_s += mask_n < prob * 0.1  # 3 -> random
                mask_s += mask_n < prob * 0.1  # 2 -> original
                mask_s += mask_n < prob * 1.00  # 1 -> [mask]
                return mask_s

            if n == 0:
                if not element_level:
                    loss_mask = torch.zeros(self.max_seq_len)
                    mask_ind = random.sample(self.Lseq, round(self.max_seq_len * mask_percent))
                    mask80 = random.sample(mask_ind, round(len(mask_ind) * 0.8))
                    left = list(set(mask_ind) - set(mask80))
                    rand10 = random.sample(left, round(len(mask_ind) * 0.1))
                    cur10 = list(set(left) - set(rand10))
                    input_ids_mask = copy.deepcopy(input_ids)
                    for i in mask80:
                        mask_word = torch.tensor(self.pianobart.mask_word_np).to(self.device)
                        input_ids_mask[i] = mask_word
                        loss_mask[i] = 1
                    for i in rand10:
                        rand_word = torch.tensor(self.pianobart.get_rand_tok()).to(self.device)
                        input_ids_mask[i] = rand_word
                        loss_mask[i] = 1
                    for i in cur10:
                        loss_mask[i] = 1
                    return input_ids_mask, loss_mask
                else:
                    loss_mask = torch.zeros(self.max_seq_len * 8)
                    mask_ind = random.sample(self.Lseq_element, round(self.max_seq_len * mask_percent * 8))
                    mask80 = random.sample(mask_ind, round(len(mask_ind) * 0.8))
                    left = list(set(mask_ind) - set(mask80))
                    rand10 = random.sample(left, round(len(mask_ind) * 0.1))
                    cur10 = list(set(left) - set(rand10))
                    input_ids_mask = copy.deepcopy(input_ids)
                    input_ids_mask = input_ids_mask.view(-1)
                    for i in mask80:
                        mask_word = torch.tensor(self.pianobart.mask_word_np).to(self.device)
                        input_ids_mask[i] = mask_word[i % 8]
                        loss_mask[i] = 1
                    for i in rand10:
                        rand_word = torch.tensor(self.pianobart.get_rand_tok()).to(self.device)
                        input_ids_mask[i] = rand_word[i % 8]
                        loss_mask[i] = 1
                    for i in cur10:
                        loss_mask[i] = 1
                    input_ids_mask = input_ids_mask.view(-1, 8)
                    loss_mask = loss_mask.view(-1, 8)
                    return input_ids_mask, loss_mask
            else:
                if n != 1:
                    # TODO
                    pass
                else:
                    max_bars = self.pianobart.n_tokens[0]
                    max_instruments = self.pianobart.n_tokens[2]
                    input_ids_mask = copy.deepcopy(input_ids)
                    if element_level:
                        loss_mask = np.zeros(self.max_seq_len * 8)
                        input_ids_mask = input_ids_mask.view(-1)
                        loss_mask[8: -8] = generate_mask((max_bars * max_instruments) * 8, mask_percent).reshape(-1, 8)[
                            ((input_ids_mask[8: -8: 8]) * max_instruments) + (
                                input_ids_mask[8 + 2: -8 + 2: 8])].flatten()
                        loss_mask = torch.tensor(loss_mask)
                        mask80 = torch.where(loss_mask == 1)[0]
                        rand10 = torch.where(loss_mask == 3)[0]
                        cur10 = torch.where(loss_mask == 2)[0]
                        for i in mask80:
                            mask_word = torch.tensor(self.pianobart.mask_word_np).to(self.device)
                            input_ids_mask[i] = mask_word[i % 8]
                            loss_mask[i] = 1
                        for i in rand10:
                            rand_word = torch.tensor(self.pianobart.get_rand_tok()).to(self.device)
                            input_ids_mask[i] = rand_word[i % 8]
                            loss_mask[i] = 1
                        for i in cur10:
                            loss_mask[i] = 1
                        input_ids_mask = input_ids_mask.view(-1, 8)
                        loss_mask = loss_mask.view(-1, 8)
                        return input_ids_mask, loss_mask
                    else:
                        loss_mask = np.zeros(self.max_seq_len)
                        loss_mask[1: -1] = generate_mask(max_bars, mask_percent)[input_ids_mask[1: -1: 1, 0]]
                        loss_mask = torch.tensor(loss_mask)
                        mask80 = torch.where(loss_mask == 1)[0]
                        rand10 = torch.where(loss_mask == 3)[0]
                        cur10 = torch.where(loss_mask == 2)[0]
                        for i in mask80:
                            mask_word = torch.tensor(self.pianobart.mask_word_np).to(self.device)
                            input_ids_mask[i] = mask_word
                            loss_mask[i] = 1
                        for i in rand10:
                            rand_word = torch.tensor(self.pianobart.get_rand_tok()).to(self.device)
                            input_ids_mask[i] = rand_word
                            loss_mask[i] = 1
                        for i in cur10:
                            loss_mask[i] = 1
                        return input_ids_mask, loss_mask

        def SentencePermutation(input_ids: torch.Tensor):
            masked = copy.deepcopy(input_ids).numpy()
            l = masked.shape[0]
            # maskedPos = [1 if i < l * mask_percent else 0 for i in range(l)]
            # length = int(l * mask_percent)
            maskedPos = [0 for i in range(l)]
            barMax = masked[-1, 0]
            sentences = dict()
            sentence = list()
            nonmasked = list()
            for i in masked:
                bar = i[0]
                if bar not in sentences:
                    sentences[bar] = list()
                sentences[bar].append(i)
                sentence.append(bar)
                nonmasked.append(i)
            sentence = list(set(sentence))
            random.shuffle(sentence)
            masked = list()
            for i in sentence:
                masked += sentences[i]
            '''print(nonmasked)
            print(masked)'''
            for i in range(len(nonmasked)):
                if (nonmasked[i] != masked[i]).any():
                    maskedPos[i] = 1
            masked = torch.from_numpy(np.array(masked))
            maskedPos = torch.from_numpy(np.array(maskedPos))
            return masked, maskedPos

        def TokenInfilling(input_ids: torch.Tensor, mask_percent, n=0, lamda=3):
            masked_tensor = torch.from_numpy(self.pianobart.mask_word_np)  # masked的token
            pad_tensor = torch.from_numpy(self.pianobart.pad_word_np)
            if n == 0:  # Octuple-level
                l = input_ids.shape[0]
                masked = torch.tensor([])
                maskpos = [0 for j in range(l)]
                for k in range(10):
                    masked = torch.tensor([])
                    maskpos = [0 for j in range(l)]
                    i = 0
                    while (i < l):
                        ran = random.random()
                        if ran < mask_percent / max(1, lamda):  # 控制期望掩蔽的数量是在mask_percent
                            p = np.random.poisson(lamda)  # 泊松采样
                            if (p == 0):  # 如果长度为1，则插入一个长度为1的mask
                                masked = torch.cat((masked, input_ids[i:i + 1]), dim=0)
                                masked = torch.cat((masked, masked_tensor.unsqueeze(0)), dim=0)
                                i += 1
                            else:  # 否则，跳过p个octuple，只插入一个长度为1的mask
                                masked = torch.cat((masked, masked_tensor.unsqueeze(0)), dim=0)
                                i += p
                        else:
                            masked = torch.cat((masked, input_ids[i:i + 1]), dim=0)
                            i += 1
                    if (masked.size()[0] <= input_ids.size()[0]):
                        for j in range(input_ids.size()[0] - masked.size()[0]):
                            masked = torch.cat((masked, pad_tensor.unsqueeze(0)), dim=0)
                        break
                    assert k < 9, "length of masked input_ids meets error in 10 rounds, please check TokenInfilling"

                # print(masked.size())
                for i in range(len(input_ids)):
                    if (input_ids[i] != masked[i]).any():
                        maskpos[i] = 1
                maskpos = torch.from_numpy(np.array(maskpos))
                return masked, maskpos
            else:  # Bar-level
                max_bars = self.pianobart.n_tokens[0]
                l = input_ids.shape[0]
                masked = copy.deepcopy(input_ids)
                maskpos = [0 for j in range(l)]
                num_mask = round(l * mask_percent)  # 应该mask的octuple数量
                cnt_bar = np.zeros(max_bars)  # 记录每个bar对应的octuple数量以控制mask数量
                bar_octuples = [[] for i in range(max_bars)]
                for i in range(l):
                    cnt_bar[input_ids[i][0]] += 1
                    bar_octuples[input_ids[i][0]].append(i)
                maskpos = torch.tensor(maskpos)
                for k in range(10):  # 确保mask后的input_ids长度没有增加,否则重新执行(概率很小)
                    masked = torch.tensor([])
                    op_pos = [0 for i in range(l)]  # 记录octuple该位置后是0:保留，1:后面填充mask,2:被删除,3:自身变成mask
                    # poisson_bar=[-1 for i in range(max_bars)]  #记录bar上的泊松采样值
                    i = 0
                    num_masked = 0
                    while (i < max_bars):
                        ran = random.random()
                        if ran < (mask_percent / max(1, lamda)):
                            p = np.random.poisson(lamda)  # 泊松采样
                            # print("i:{}  p:{}".format(i,p))
                            if (p == 0):
                                if (cnt_bar[i] != 0):
                                    op_pos[bar_octuples[i][-1]] = 1  # 该bar最后一个octuple后填充mask
                                i += 1
                            else:
                                cur_num_mask = sum(cnt_bar[i:min(i + p, max_bars)])  # 当前牵涉到的mask的octuple数量
                                if ((num_masked + cur_num_mask) <= num_mask):  # 控制mask的octuple数量
                                    num_masked += cur_num_mask
                                    first_bar = True
                                    for j in range(i, min(i + p, max_bars)):
                                        for k in bar_octuples[j]:
                                            op_pos[k] = 2
                                        if ((cnt_bar[j] != 0) & first_bar):
                                            first_bar = False
                                            op_pos[bar_octuples[j][0]] = 3  # 首个非空bar的第1个octuple进行掩码
                                    i += p
                                else:
                                    i += 1

                        else:
                            i += 1
                    '''print("op_pos:{}".format(op_pos))
                    print("0: keep origin  1: add mask behind 2: delete  3: mask (line : 420)")'''
                    i = 0
                    while (i < l):
                        if (op_pos[i] == 0):
                            masked = torch.cat((masked, input_ids[i:i + 1]), dim=0)
                        elif op_pos[i] == 1:  # 后面增加一个mask
                            masked = torch.cat((masked, input_ids[i:i + 1]), dim=0)
                            masked = torch.cat((masked, masked_tensor.unsqueeze(0)), dim=0)
                        elif op_pos[i] == 2:  # 删除
                            pass
                        else:  # 自身mask
                            masked = torch.cat((masked, masked_tensor.unsqueeze(0)), dim=0)
                        i += 1
                    if (masked.size()[0] <= input_ids.size()[0]):  # 填充pad
                        for j in range(input_ids.size()[0] - masked.size()[0]):
                            masked = torch.cat((masked, pad_tensor.unsqueeze(0)), dim=0)
                        break
                    assert k < 9, "length of masked input_ids meets error in 10 rounds, please check TokenInfilling"
                for i in range(len(input_ids)):  # 求maskpos
                    if (input_ids[i] != masked[i]).any():
                        maskpos[i] = 1
                maskpos = torch.from_numpy(np.array(maskpos))
                return masked, maskpos

        def DocumentRotation(input_ids: torch.Tensor):
            l = input_ids.shape[0]
            ran = random.randint(0, l - 1)
            masked = torch.cat((input_ids[ran:], input_ids[0:ran]), dim=0)
            if ran != 0:
                maskpos = [1 for j in range(l)]
            else:
                maskpos = [0 for j in range(l)]
            maskpos = torch.from_numpy(np.array(maskpos))
            return masked, maskpos

        if choice is None:
            choice = random.randint(1, 5)
            # choice = 1
        # print(f'choice = {choice}')
        if choice == 1:
            return TokenDeletion(input_ids, self.mask_percent, self.pianobart.pad_word_np)
        elif choice == 2:
            n = random.randint(0, 1)
            element_level = (random.randint(0, 1) == 0)
            return TokenMask(input_ids, self.mask_percent, n, element_level)
        elif choice == 3:
            # ASSERTED TRIGGER
            # IndexError: too many indices for tensor of dimension 2
            return SentencePermutation(input_ids)
        elif choice == 4:
            n = random.randint(0, 1)
            return TokenInfilling(input_ids, self.mask_percent, n=n)
        elif choice == 5:
            return DocumentRotation(input_ids)

def load_data_pretrain(datasets,mode):
    if mode=="pretrain":
        to_concat = []
        root = 'Data/output'

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

# test
if __name__ == '__main__':
    with open('./Data/Octuple.pkl', 'rb') as f:
        e2w, w2e = pickle.load(f)
    pianobart = PianoBart(bartConfig=BartConfig(max_position_embeddings=10, d_model=16), e2w=e2w, w2e=w2e)
    p = Pretrainer(pianobart, None, None, 0.01, None, 10, 0.5, True, None)
    print("MASK", pianobart.mask_word_np)

    test_TokenDeletion = False
    # test for TokenDeletion
    if test_TokenDeletion:
        input_ids = list()
        for i in range(10):
            tmp = [j for j in range(8 * i, 8 * (i + 1))]
            input_ids.append(tmp)
        input_ids = torch.tensor(input_ids)
        print("input\n", input_ids)
        print("\ntest for TokenDeletion")
        input_mask, mask_pos = p.gen_mask(input_ids, 1)
        print(input_mask)
        print(mask_pos)
        if mask_pos.size()[-1] != 8:
            mask_pos = np.repeat(mask_pos[:, np.newaxis], 8, axis=1)
            print(mask_pos)

    # test for TokenMask
    test_TokenMask = False
    if test_TokenMask:
        input_ids = list()
        for i in range(10):
            tmp = [j for j in range(8 * i, 8 * (i + 1))]
            if i < 5:
                tmp[0] = 0
                tmp[2] = 0
            else:
                tmp[0] = 100
                tmp[2] = 100
            input_ids.append(tmp)
        input_ids = torch.tensor(input_ids)
        print("input\n", input_ids)
        print("\ntest for TokenMask")
        input_mask, mask_pos = p.gen_mask(input_ids, 2)
        print(input_mask)
        print(mask_pos)
        if mask_pos.size()[-1] != 8:
            mask_pos = np.repeat(mask_pos[:, np.newaxis], 8, axis=1)
            print(mask_pos)

    # test for SentencePermutation
    test_SentencePermutation = False
    if test_SentencePermutation:
        input_ids = list()
        for i in range(12):
            tmp = [j for j in range(8 * i, 8 * (i + 1))]
            if i < 5:
                # tmp[0]=i//4
                tmp[2] = 0
            else:
                # tmp[0]=100
                tmp[2] = 100
            tmp[0] = i // 4
            input_ids.append(tmp)
        input_ids = torch.tensor(input_ids)
        print("input\n", input_ids)
        print("\ntest for SentencePermutation")
        input_mask, mask_pos = p.gen_mask(input_ids, 3)
        print(input_mask)
        print(mask_pos)
        if mask_pos.size()[-1] != 8:
            mask_pos = np.repeat(mask_pos[:, np.newaxis], 8, axis=1)
            print(mask_pos)

    # test for SentencePermutation
    test_TokenInfilling = False
    if test_TokenInfilling:
        input_ids = list()
        for i in range(8):
            tmp = [j for j in range(8 * i, 8 * (i + 1))]  # 增加一些相邻的bar
            tmp2 = [j for j in range(8 * i + 1, 8 * (i + 1) + 1)]
            input_ids.append(tmp)
            input_ids.append(tmp)
            input_ids.append(tmp2)
        input_ids = torch.tensor(input_ids)
        print("input\n", input_ids)
        print("\ntest for TokenInfilling")
        input_mask, mask_pos = p.gen_mask(input_ids, 4)
        print(input_mask)
        print(mask_pos)
        if mask_pos.size()[-1] != 8:
            mask_pos = np.repeat(mask_pos[:, np.newaxis], 8, axis=1)
            print(mask_pos)

    # test for DocumentRotation
    test_DocumentRotation = False
    if test_DocumentRotation:
        input_ids = list()
        for i in range(10):
            tmp = [j for j in range(8 * i, 8 * (i + 1))]
            input_ids.append(tmp)
        input_ids = torch.tensor(input_ids)
        print("input\n", input_ids)
        print("\ntest for DocumentRotation")
        input_mask, mask_pos = p.gen_mask(input_ids, 5)
        print(input_mask)
        print(mask_pos)
        if mask_pos.size()[-1] != 8:
            mask_pos = np.repeat(mask_pos[:, np.newaxis], 8, axis=1)
            print(mask_pos)
