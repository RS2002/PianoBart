import shutil
import numpy as np
import tqdm
import torch
import torch.nn as nn
from transformers import AdamW
import os
import argparse
from model import TokenClassification, SequenceClassification
# from torch.nn.utils import clip_grad_norm_


def get_args_finetune():
    parser = argparse.ArgumentParser(description='')

    ### mode ###
    parser.add_argument(
        '--task', choices=['melody', 'velocity', 'composer', 'emotion'], required=True)
    ### dataset & data root ###
    parser.add_argument(
        '--dataset', type=str, choices=('asap', 'Pianist8', 'POP909', 'EMOPIA', 'GiantMIDI1k'), required=True)
    parser.add_argument('--dataroot', type=str, default=None)
    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='./Data/Octuple.pkl')
    parser.add_argument('--name', type=str, default='pianobart')
    parser.add_argument(
        '--ckpt', default='result/pretrain/pianobart/model_best.ckpt')

    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--class_num', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=1024)
    # layer nums of encoder & decoder
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--ffn_dims', type=int, default=2048)  # FFN dims
    parser.add_argument('--heads', type=int, default=8)  # attention heads

    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='initial learning rate')
    parser.add_argument('--nopretrain', action="store_true")  # default: false

    ### cuda ###
    parser.add_argument("--cpu", action="store_true")  # default=False
    parser.add_argument("--cuda_devices", type=int, nargs='+',
                        default=[6, 7], help="CUDA device ids")

    parser.add_argument("--error_correction",
                        action="store_true")  # default: false

    args = parser.parse_args()

    # check args
    if args.class_num is None:
        if args.task == 'melody':
            args.class_num = 4
        elif args.task == 'velocity':
            args.class_num = 7
        elif args.task == 'composer':
            args.class_num = 8
        elif args.task == 'emotion':
            args.class_num = 4

    return args


class FinetuneTrainer:
    def __init__(self, pianobart, train_dataloader, valid_dataloader, test_dataloader,
                 lr, class_num, hs, testset_shape, cpu, cuda_devices=None, model=None, SeqClass=False, error=False):

        device_name = "cuda"
        if cuda_devices is not None and len(cuda_devices) >= 1:
            device_name += ":" + str(cuda_devices[0])
        self.device = torch.device(
            device_name if torch.cuda.is_available() and not cpu else 'cpu')
        print('   device:', self.device)
        self.pianobart = pianobart
        self.SeqClass = SeqClass
        self.class_num = class_num
        if model != None:  # load model
            print('load a fine-tuned model')
            self.model = model.to(self.device)
        else:
            print('init a fine-tune model, sequence-level task?', SeqClass)
            if SeqClass:
                self.model = SequenceClassification(
                    self.pianobart, class_num, hs).to(self.device)
            else:
                self.model = TokenClassification(
                    self.pianobart, class_num+1, hs).to(self.device)

        #        for name, param in self.model.named_parameters():
        #            if 'midibert.bert' in name:
        #                    param.requires_grad = False
        #            print(name, param.requires_grad)

        if len(cuda_devices) > 1 and not cpu:
            print("Use %d GPUS" % len(cuda_devices))
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        elif (len(cuda_devices) == 1 or torch.cuda.is_available()) and not cpu:
            print("Use GPU", end=" ")
            print(self.device)
        else:
            print("Use CPU")

        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        self.test_data = test_dataloader

        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

        self.testset_shape = testset_shape if not error else testset_shape[:-1]

        # print(self.testset_shape)

    def compute_loss(self, predict, target, loss_mask, seq):
        loss = self.loss_func(predict, target)
        if not seq:
            loss = loss * loss_mask
            loss = torch.sum(loss) / torch.sum(loss_mask)
        else:
            loss = torch.sum(loss) / loss.shape[0]
        return loss

    def train(self):
        self.model.train()
        train_loss, train_acc = self.iteration(
            self.train_data, 0, self.SeqClass)
        return train_loss, train_acc

    def valid(self):
        self.model.eval()
        valid_loss, valid_acc = self.iteration(
            self.valid_data, 1, self.SeqClass)
        return valid_loss, valid_acc

    def test(self):
        self.model.eval()
        test_loss, test_acc, all_output = self.iteration(
            self.test_data, 2, self.SeqClass)
        return test_loss, test_acc, all_output

    def iteration(self, training_data, mode, seq):
        pbar = tqdm.tqdm(training_data, disable=False)

        total_acc, total_cnt, total_loss = 0, 0, 0

        if mode == 0:
            self.model.train()
            torch.set_grad_enabled(True)
        else:
            self.model.eval()
            torch.set_grad_enabled(False)

        if mode == 2:  # testing
            all_output = torch.empty(self.testset_shape)
            cnt = 0

        for x, y in pbar:  # (batch, 512, 768)
            batch = x.shape[0]
            # seq: (batch, 512, 4), (batch) / token: , (batch, 512)
            x, y = x.to(self.device), y.to(self.device)

            x = x.long()
            y = y.long()
            # y=y.squeeze()
            # Remove the last dimension
            y = torch.squeeze(y, dim=-1)
            # print(y.shape)

            # avoid attend to pad word
            attn = (x[:, :, 0] != self.pianobart.bar_pad_word).float().to(
                self.device)  # (batch, seq_len)

            if seq:
                # seq: (batch, class_num) / token: (batch, 512, class_num)
                y_hat = self.model.forward(
                    input_ids_encoder=x, encoder_attention_mask=attn)
            else:
                # class_num表示pad对应的token
                y_shift = torch.zeros_like(y)+self.class_num
                attn_shift = torch.zeros_like(attn)
                y_shift[:, 1:] = y[:, :-1]
                attn_shift[:, 1:] = attn[:, :-1]
                attn_shift[:, 0] = attn[:, 0]
                y_hat = self.model.forward(input_ids_encoder=x, input_ids_decoder=y_shift,
                                           encoder_attention_mask=attn, decoder_attention_mask=attn_shift)

            # get the most likely choice with max
            output = np.argmax(y_hat.cpu().detach().numpy(), axis=-1)
            output = torch.from_numpy(output).to(self.device)
            if mode == 2:
                all_output[cnt: cnt + batch] = output
                cnt += batch

            # accuracy
            if not seq:
                acc = torch.sum((y == output).float() * attn)
                total_acc += acc
                total_cnt += torch.sum(attn).item()
            else:
                acc = torch.sum((y == output).float())
                total_acc += acc
                total_cnt += y.shape[0]

            # calculate losses
            if not seq:
                y_hat = y_hat.permute(0, 2, 1)
            loss = self.compute_loss(y_hat, y, attn, seq)
            total_loss += loss

            # udpate only in train
            if mode == 0:
                self.model.zero_grad()
                loss.backward()
                self.optim.step()

        if mode == 2:
            return round(total_loss.item() / len(training_data), 4), round(total_acc.item() / total_cnt, 4), all_output
        return round(total_loss.item() / len(training_data), 4), round(total_acc.item() / total_cnt, 4)

    def save_checkpoint(self, epoch, train_acc, valid_acc,
                        valid_loss, train_loss, is_best, filename):
        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
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


def load_data_finetune(dataset, task, data_root=None):
    if data_root is None:
        data_root = 'Data/finetune/others'

    if dataset == 'emotion':
        dataset = 'emopia'

    if dataset not in ['POP909', 'pop909', 'composer', 'EMOPIA', 'asap', 'Pianist8', 'maestro', 'GiantMIDI1k']:
        print(f'Dataset {dataset} not supported')
        exit(1)

    if task == "gen":
        X_train = np.load(os.path.join(
            data_root, f'{dataset}_train.npy'), allow_pickle=True)
        X_val = np.load(os.path.join(
            data_root, f'{dataset}_valid.npy'), allow_pickle=True)
        X_test = np.load(os.path.join(
            data_root, f'{dataset}_test.npy'), allow_pickle=True)

        print('X_train: {}, X_valid: {}, X_test: {}'.format(
            X_train.shape, X_val.shape, X_test.shape))
        y_train = np.load(os.path.join(
            data_root, f'{dataset}_train_genans.npy'), allow_pickle=True)
        y_val = np.load(os.path.join(
            data_root, f'{dataset}_valid_genans.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(
            data_root, f'{dataset}_test_genans.npy'), allow_pickle=True)
    else:
        X_train = np.load(os.path.join(
            data_root, f'{dataset}_train.npy'), allow_pickle=True)
        X_val = np.load(os.path.join(
            data_root, f'{dataset}_valid.npy'), allow_pickle=True)
        X_test = np.load(os.path.join(
            data_root, f'{dataset}_test.npy'), allow_pickle=True)

        print('X_train: {}, X_valid: {}, X_test: {}'.format(
            X_train.shape, X_val.shape, X_test.shape))
        if dataset == 'pop909':
            y_train = np.load(os.path.join(
                data_root, f'{dataset}_train_{task[:3]}ans.npy'), allow_pickle=True)
            y_val = np.load(os.path.join(
                data_root, f'{dataset}_valid_{task[:3]}ans.npy'), allow_pickle=True)
            y_test = np.load(os.path.join(
                data_root, f'{dataset}_test_{task[:3]}ans.npy'), allow_pickle=True)
        else:
            y_train = np.load(os.path.join(
                data_root, f'{dataset}_train_ans.npy'), allow_pickle=True)
            y_val = np.load(os.path.join(
                data_root, f'{dataset}_valid_ans.npy'), allow_pickle=True)
            y_test = np.load(os.path.join(
                data_root, f'{dataset}_test_ans.npy'), allow_pickle=True)

    print('y_train: {}, y_valid: {}, y_test: {}'.format(
        y_train.shape, y_val.shape, y_test.shape))

    return X_train, X_val, X_test, y_train, y_val, y_test
