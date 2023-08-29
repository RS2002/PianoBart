from transformers import AdamW
import torch
import torch.nn as nn
from model import PianoBartLM
import argparse
import shutil
import numpy as np
import tqdm
from torch.nn.utils import clip_grad_norm_
import sys
import os


def get_args_ablation():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("--datasets", type=str, nargs='+', default=['maestro'])

    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='./Data/Octuple.pkl')
    parser.add_argument('--name', type=str, default='pianobart')
    parser.add_argument('--ckpt', default='result/pretrain/pianobart/model_best.ckpt')

    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_seq_len', type=int, default=1024, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=1024)
    parser.add_argument('--layers', type=int, default=8)  # layer nums of encoder & decoder
    parser.add_argument('--ffn_dims', type=int, default=2048)  # FFN dims
    parser.add_argument('--heads', type=int, default=8)  # attention heads

    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
    parser.add_argument('--nopretrain', action="store_true")  # default: false

    ### cuda ###
    parser.add_argument("--cpu", action="store_true")  # default=False
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0, 1, 2, 3], help="CUDA device ids")

    args = parser.parse_args()

    return args


class AblationTrainer:
    def __init__(self, pianobart, train_dataloader, valid_dataloader, test_dataloader,
                 lr, cpu, cuda_devices=None, model=None):
        device_name = "cuda"
        if cuda_devices is not None and len(cuda_devices) >= 1:
            device_name += ":" + str(cuda_devices[0])
        self.device = torch.device(device_name if torch.cuda.is_available() and not cpu else 'cpu')
        print('   device:', self.device)
        self.pianobart = pianobart
        if model != None:  # load model
            print('load a fine-tuned model')
            self.model = model.to(self.device)
        else:
            print('init a fine-tune model')
            self.model = PianoBartLM(pianobart).to(self.device)

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
        self.test_data = test_dataloader


        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.loss_func = nn.NLLLoss()
        #self.loss_func = nn.CrossEntropyLoss(reduction='none')



    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        #TODO: add weights for different attributes
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def train(self):
        self.model.train()
        train_loss, train_acc = self.iteration(self.train_data, 0)
        return train_loss, train_acc

    def valid(self):
        self.model.eval()
        valid_loss, valid_acc = self.iteration(self.valid_data, 1)
        return valid_loss, valid_acc

    def test(self):
        self.model.eval()
        test_loss, test_acc, all_output = self.iteration(self.test_data, 2)
        return test_loss, test_acc, all_output

    def iteration(self, training_data, mode):
        pbar = tqdm.tqdm(training_data, disable=False)

        total_acc, total_loss = [0] * len(self.pianobart.e2w), 0
        total_cnt = 0

        if mode ==0:
            self.model.train()
            torch.set_grad_enabled(True)
        else:
            self.model.eval()
            torch.set_grad_enabled(False)

        if mode == 2:  # testing
            all_output = torch.empty(self.testset_shape)
            cnt = 0

        for ori_seq_batch in pbar:
            batch = ori_seq_batch.shape[0]
            ori_seq_batch = ori_seq_batch.to(self.device)
            shift_seq_batch = torch.zeros_like(ori_seq_batch)
            shift_seq_batch[:,1:,:]=ori_seq_batch[:,:-1,:]
            shift_seq_batch[:,0,:]=torch.tensor(self.pianobart.sos_word_np)
            shift_seq_batch.to(self.device)
            length = (ori_seq_batch[:, :, 0] != self.pianobart.bar_pad_word).int()
            loss_mask = torch.zeros(batch, 1024)
            for i in range(batch):
                ori_seq_batch[i,length//2:,:]=self.pianobart.pad_word_np
                loss_mask[i,length//2+1:length+1]=1
            attn_encoder = (ori_seq_batch[:, :, 0] != self.pianobart.bar_pad_word).float().to(self.device)
            attn_decoder = (shift_seq_batch[:, :, 0] != self.pianobart.bar_pad_word).float().to(self.device)
            y_hat = self.model.forward(input_ids_encoder=ori_seq_batch, input_ids_decoder=shift_seq_batch, encoder_attention_mask=attn_encoder,decoder_attention_mask=attn_decoder)
            # get the most likely choice with max
            outputs = []
            for i, etype in enumerate(self.pianobart.e2w):
                output = np.argmax(y_hat[i].cpu().detach().numpy(), axis=-1)
                outputs.append(output)
            outputs = np.stack(outputs, axis=-1)
            outputs = torch.from_numpy(outputs).to(self.device)  # (batch, seq_len)

            if mode == 2:
                all_output[cnt: cnt + batch] = outputs
                cnt += batch

            all_acc = []
            for i in range(8):
                acc = torch.sum((shift_seq_batch[:, :, i] == outputs[:, :, i]).float())
                acc /= torch.sum(loss_mask)
                all_acc.append(acc)
            total_acc = [sum(x) for x in zip(total_acc, all_acc)]
            total_cnt += torch.sum(attn_decoder).item()

            # reshape (b, s, f) -> (b, f, s)
            for i, etype in enumerate(self.pianobart.e2w):
                # print('before',y[i][:,...].shape)   # each: (4,512,5), (4,512,20), (4,512,90), (4,512,68)
                y_hat[i] = y_hat[i][:, ...].permute(0, 2, 1)  # 维度交换

            # calculate losses
            losses, n_tok = [], []
            for i, etype in enumerate(self.pianobart.e2w):
                n_tok.append(len(self.pianobart.e2w[etype]))
                losses.append(self.compute_loss(y_hat[i], shift_seq_batch[..., i], loss_mask))
            total_loss_all = [x * y for x, y in zip(losses, n_tok)]
            loss = sum(total_loss_all) / sum(n_tok)  # weighted
            total_loss += loss.item()

            # update only in train
            if mode==0:
                self.model.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 3.0)  # 用于裁剪梯度，防止梯度爆炸
                self.optim.step()

            accs = list(map(float, all_acc))
            sys.stdout.write(
                'Loss: {:06f} | loss: {:03f}, {:03f}, {:03f}, {:03f}, {:03f}, {:03f}, {:03f}, {:03f}\n'.format(
                    loss, *losses))
            sys.stdout.write(
                'Acc: {:06f} | acc: {:03f}, {:03f}, {:03f}, {:03f}, {:03f}, {:03f}, {:03f}, {:03f}\n'.format(
                    np.average(accs), *accs))

        if mode == 2:
            return round(total_loss / len(training_data), 4), round(np.average(total_acc) / total_cnt, 4), all_output
        return round(total_loss / len(training_data), 4), round(np.average(total_acc)/ total_cnt, 4)



    def save_checkpoint(self, epoch, train_acc, valid_acc,
                        valid_loss, train_loss, is_best, filename):
        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'valid_acc': valid_acc,
            'valid_loss': valid_loss,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'optimizer': self.optim.state_dict()
        }
        torch.save(state, filename)

        best_mdl = filename.split('.')[0] + '_best.ckpt'

        if is_best:
            shutil.copyfile(filename, best_mdl)


def load_data_ablation(datasets,mode):
    if mode=="ablation":
        to_concat = []
        root = 'Data/output_generation/pretrain_method'

        # for dataset in datasets:
        #     data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)
        #     print(f'   {dataset}: {data.shape}')
        #     to_concat.append(data)
        for dataset in datasets:
            data_train = np.load(os.path.join(root, dataset, 'train.npy'), allow_pickle = True)
            data_test = np.load(os.path.join(root, dataset, 'test.npy'), allow_pickle = True)
            data_valid = np.load(os.path.join(root, dataset, 'valid.npy'), allow_pickle = True)
            data = np.concatenate((data_train, data_test, data_valid), axis = 0)
            print(f'   {dataset}: {data.shape}')
            to_concat.append(data)

        training_data = np.vstack(to_concat)
        print('   > all training data:', training_data.shape)
        # shuffle during training phase
        index = np.arange(len(training_data))
        np.random.shuffle(index)
        training_data = training_data[index]
        split1 = int(len(training_data) * 0.8)
        split2 = int(len(training_data) * 0.9)
        X_train, X_val, X_test = training_data[:split1], training_data[split1:split2], training_data[split2:]
        return X_train, X_val, X_test

    else:
        return None