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

class Pretrainer:
    def __init__(self, pianobart: PianoBart, train_dataloader, valid_dataloader,
                 lr, batch, max_seq_len, mask_percent, cpu, cuda_devices=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else 'cpu')
        self.pianobart = pianobart  # save this for ckpt
        self.model = PianoBartLM(pianobart).to(self.device)
        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('# total parameters:', self.total_params)

        if torch.cuda.device_count() > 1 and not cpu:
            print("Use %d GPUS" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_data = train_dataloader
        self.valid_data = valid_dataloader

        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.batch = batch
        self.max_seq_len = max_seq_len
        self.mask_percent = mask_percent
        self.Lseq = [i for i in range(self.max_seq_len)]
        self.Lseq_element = [i for i in range(self.max_seq_len*8)]
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
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def iteration(self, training_data, max_seq_len, train=True):
        pbar = tqdm.tqdm(training_data, disable=False)  # 显示进度条

        total_acc, total_losses = [0] * len(self.pianobart.e2w), 0

        for ori_seq_batch in pbar:
            batch = ori_seq_batch.shape[0]
            ori_seq_batch = ori_seq_batch.to(self.device)  # (batch, seq_len, 8)
            input_ids_encoder = copy.deepcopy(ori_seq_batch)
            input_ids_decoder = torch.zeros_like(input_ids_encoder)
            loss_mask = torch.zeros(batch, max_seq_len,8)
            for b in range(batch):
                shifted_input_ids = input_ids_encoder[b].new_zeros(input_ids_encoder[b].shape)
                shifted_input_ids[:, 1:] = input_ids_encoder[b][:, :-1].clone()
                shifted_input_ids[:, 0] = torch.tensor(self.pianobart.sos_word_np)
                input_ids_decoder[b]=shifted_input_ids
                input_mask, mask_pos=self.gen_mask(input_ids_encoder[b])
                if mask_pos.size()[-1] != 8:
                    mask_pos = np.repeat(mask_pos[:, np.newaxis], 8, axis=1)
                input_ids_encoder[b]=input_mask
                loss_mask[b]=mask_pos

            loss_mask = loss_mask.to(self.device)
            # avoid attend to pad word
            encoder_attention_mask = (input_ids_encoder[:, :, 0] != self.pianobart.bar_pad_word).float().to(self.device)  # (batch, seq_len)
            decoder_attention_mask = (input_ids_decoder[:, :, 0] != self.pianobart.bar_pad_word).float().to(self.device)

            y = self.model.forward(input_ids_encoder, input_ids_decoder, encoder_attention_mask, decoder_attention_mask)

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
                acc = torch.sum((ori_seq_batch[:, :, i] == outputs[:, :, i]).float() * loss_mask[:,:,i])
                acc /= torch.sum(loss_mask[:,:,i])
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
                losses.append(self.compute_loss(y[i], ori_seq_batch[..., i], loss_mask[:,:,i]))
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
                'Loss: {:06f} | loss: {:03f}, {:03f}, {:03f}, {:03f} | acc: {:03f}, {:03f}, {:03f}, {:03f} \r'.format(
                    total_loss, *losses, *accs))

            losses = list(map(float, losses))
            total_losses += total_loss.item()

        return round(total_losses / len(training_data), 3), [round(x.item() / len(training_data), 3) for x in total_acc]

    def gen_mask(self, input_ids: torch.Tensor,choice=None):
        # for now, n is only used to represent to delete or masked
        # if n == -1, delete
        # else masked
        # TODO
        # more detailed mask, like mask 1/8, 1/4, 1/2, 1/1 (1/n)
        def TokenDeletion(input_ids: torch.Tensor, mask_percent, replacement: np.array, n = -1):
            def deleteOctuple(input_ids: torch.Tensor, mask_percent, replacement: torch.Tensor):
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
                barMax = masked[-1,0]
                print(barMax)
                length = int(barMax * mask_percent)
                maskBarPos = [1 if i < length else 0 for i in range(barMax)]
                random.shuffle(maskBarPos)
                maskBarPos = np.array(maskBarPos)
                count = 0
                for i in range(len(masked)):
                    if maskBarPos[masked[i,0]] == 1:
                        masked = np.delete(masked, i - count, axis = 0)
                        count += 1
                for i in range(length):
                    masked = np.append(masked, replacement.reshape(1, 8), axis = 0)
                return torch.from_numpy(masked), torch.from_numpy(maskBarPos)

        # for now, n is only used to represent to Octuple-level mask or Bar-level mask
        # if n == 0, Octuple-level mask
        # else Bar-level mask
        # if element_level=True, element-level mask
        # else Octuple-level mask
        # TODO
        # more detailed mask, like mask 1/8, 1/4, 1/2, 1/1 (1/n) bar
        def TokenMask(input_ids: torch.Tensor, mask_percent,n=-1,element_level=False):
            def generate_mask(sz, prob):
                mask_n = np.random.rand(sz)
                mask_s = np.zeros(sz, dtype=np.int8)
                mask_s += mask_n < prob * 0.1  # 3 -> random
                mask_s += mask_n < prob * 0.1 # 2 -> original
                mask_s += mask_n < prob * 1.00  # 1 -> [mask]
                return mask_s
            if n==0:
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
                    loss_mask = torch.zeros(self.max_seq_len*8)
                    mask_ind = random.sample(self.Lseq_element, round(self.max_seq_len * mask_percent*8))
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
                    input_ids_mask=input_ids_mask.view(-1,8)
                    loss_mask=loss_mask.view(-1,8)
                    return input_ids_mask, loss_mask
            else:
                if n!=1:
                    #TODO
                    pass
                else:
                    max_bars=self.pianobart.n_tokens[0]
                    max_instruments=self.pianobart.n_tokens[2]
                    input_ids_mask = copy.deepcopy(input_ids)
                    if element_level:
                        loss_mask = np.zeros(self.max_seq_len * 8)
                        input_ids_mask = input_ids_mask.view(-1)
                        loss_mask[8: -8] = generate_mask((max_bars * max_instruments) * 8, mask_percent).reshape(-1, 8)[((input_ids_mask[8: -8: 8]) * max_instruments) + (input_ids_mask[8 + 2: -8 + 2: 8])].flatten()
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
                        loss_mask[1: -1] = generate_mask(max_bars, mask_percent)[input_ids_mask[1: -1: 1,0]]
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

        # TODO(choice 3~5)
        if choice is None:
            choice=random.randint(1,5)
        if choice==1:
            return TokenDeletion(input_ids, self.mask_percent, self.pianobart.pad_word_np)
        elif choice==2:
            n = random.randint(0,1)
            element_level = (random.randint(0,1)==0)
            return TokenMask(input_ids, self.mask_percent,n,element_level)
        elif choice==3:
            pass
        elif choice==4:
            pass
        elif choice==5:
            pass


#test
if __name__ == '__main__':
    with open('./Data/Octuple.pkl', 'rb') as f:
        e2w, w2e = pickle.load(f)
    pianobart = PianoBart(bartConfig=BartConfig(max_position_embeddings=10, d_model=16), e2w=e2w, w2e=w2e)
    p = Pretrainer(pianobart, None, None, 0.01, None, 10, 0.5, True, None)
    print("MASK",pianobart.mask_word_np)

    test_TokenDeletion=False
    # test for TokenDeletion
    if test_TokenDeletion:
        input_ids = list()
        for i in range(10):
            tmp = [j for j in range(8 * i, 8 * (i + 1))]
            input_ids.append(tmp)
        input_ids = torch.tensor(input_ids)
        print("input\n", input_ids)
        print("\ntest for TokenDeletion")
        input_mask, mask_pos=p.gen_mask(input_ids,1)
        print(input_mask)
        print(mask_pos)
        if mask_pos.size()[-1] != 8:
            mask_pos = np.repeat(mask_pos[:, np.newaxis], 8, axis=1)
            print(mask_pos)

    test_TokenMask = False
    if test_TokenMask:
        input_ids = list()
        for i in range(10):
            tmp = [j for j in range(8 * i, 8 * (i + 1))]
            if i<5:
                tmp[0]=0
                tmp[2]=0
            else:
                tmp[0]=100
                tmp[2]=100
            input_ids.append(tmp)
        input_ids = torch.tensor(input_ids)
        print("input\n", input_ids)
        # test for TokenMask
        print("\ntest for TokenMask")
        input_mask, mask_pos = p.gen_mask(input_ids, 2)
        print(input_mask)
        print(mask_pos)
        if mask_pos.size()[-1] != 8:
            mask_pos = np.repeat(mask_pos[:, np.newaxis], 8, axis=1)
            print(mask_pos)
