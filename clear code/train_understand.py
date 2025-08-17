import copy
import shutil
import numpy as np
import tqdm
import torch
import torch.nn as nn
from transformers import AdamW
import os
import argparse
from model import TokenClassification, SequenceClassification
from torch.nn.utils import clip_grad_norm_
import random
import pickle
from dataset import FinetuneDataset
from torch.utils.data import DataLoader
from transformers import BartConfig
from PianoBart import PianoBart

def get_args_finetune():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--seq', action="store_true",default=False)
    parser.add_argument('--class_num', type=int,  required=True)
    parser.add_argument('--dataset', type=str,  required=True)
    parser.add_argument('--dataroot', type=str,  required=True)
    parser.add_argument('--dict_file', type=str, default='./Data/Octuple.pkl')
    parser.add_argument('--ckpt', default='./pianobart.ckpt')

    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_seq_len', type=int, default=1024, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=1024)
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--ffn_dims', type=int, default=2048)
    parser.add_argument('--heads', type=int, default=8)

    parser.add_argument('--epochs', type=int, default=50,help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5,help='initial learning rate')
    parser.add_argument('--nopretrain', action="store_true",default=False)

    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0], help="CUDA device ids")

    parser.add_argument("--weight", type=float, default=None, help="weight of regularization")

    parser.add_argument("--error_correction",action="store_true",default=False)

    args = parser.parse_args()

    return args


def load_data_finetune(dataset, data_root):
    X_train = np.load(os.path.join(
        data_root, f'{dataset}_train.npy'), allow_pickle=True)
    X_val = np.load(os.path.join(
        data_root, f'{dataset}_valid.npy'), allow_pickle=True)
    X_test = np.load(os.path.join(
        data_root, f'{dataset}_test.npy'), allow_pickle=True)
    print('X_train: {}, X_valid: {}, X_test: {}'.format(
        X_train.shape, X_val.shape, X_test.shape))

    y_train = np.load(os.path.join(
        data_root, f'{dataset}_train_ans.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(
        data_root, f'{dataset}_valid_ans.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(
        data_root, f'{dataset}_test_ans.npy'), allow_pickle=True)
    print('y_train: {}, y_valid: {}, y_test: {}'.format(
        y_train.shape, y_val.shape, y_test.shape))

    return X_train, X_val, X_test, y_train, y_val, y_test

class FinetuneTrainer:
    def __init__(self, pianobart, train_dataloader, valid_dataloader, test_dataloader,
                 lr, class_num, hs, testset_shape, cpu, cuda_devices=None, model=None, SeqClass=False, error=False, weight=None):

        device_name = "cuda"
        if cuda_devices is not None and len(cuda_devices) >= 1:
            device_name += ":" + str(cuda_devices[0])
        self.device = torch.device(
            device_name if torch.cuda.is_available() and not cpu else 'cpu')
        print('   device:', self.device)
        self.pianobart = pianobart
        self.SeqClass = SeqClass
        self.class_num = class_num

        if model is not None:
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
        self.weight = weight
        self.error = error

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
        torch.set_grad_enabled(True)
        train_loss, train_acc = self.iteration(self.train_data, 0, self.SeqClass)
        return train_loss, train_acc

    def valid(self):
        self.model.eval()
        torch.set_grad_enabled(False)
        valid_loss, valid_acc = self.iteration(self.valid_data, 1, self.SeqClass)
        return valid_loss, valid_acc

    def test(self):
        self.model.eval()
        torch.set_grad_enabled(False)
        test_loss, test_acc, all_output = self.iteration(self.test_data, 2, self.SeqClass)
        return test_loss, test_acc, all_output

    def iteration(self, training_data, mode, seq):
        pbar = tqdm.tqdm(training_data, disable=False)

        total_acc, total_cnt, total_loss = 0, 0, 0

        if mode == 2:
            all_output = torch.empty(self.testset_shape)
            cnt = 0

        for x, y in pbar:
            batch = x.shape[0]
            x, y = x.to(self.device), y.to(self.device)
            x = x.long()
            y = y.long()
            if self.error:
                y = torch.squeeze(y, dim=-1)

            attn = (x[:, :, 0] != self.pianobart.bar_pad_word).float().to(self.device)

            if seq:
                y_hat = self.model.forward(input_ids_encoder=x, encoder_attention_mask=attn)
            else:
                # y_shift = torch.zeros_like(y)+self.class_num
                # y_shift[:, 1:] = y[:, :-1]
                # attn_shift = torch.zeros_like(attn)
                # attn_shift[:, 1:] = attn[:, :-1]
                # attn_shift[:, 0] = attn[:, 0]

                y_shift = copy.deepcopy(x).to(self.device)
                attn_shift = copy.deepcopy(attn).to(self.device)

                y_hat = self.model.forward(input_ids_encoder=x, input_ids_decoder=y_shift,encoder_attention_mask=attn, decoder_attention_mask=attn_shift)

            output = np.argmax(y_hat.cpu().detach().numpy(), axis=-1)
            output = torch.from_numpy(output).to(self.device)
            if mode == 2:
                all_output[cnt: cnt + batch] = output
                cnt += batch

            if not seq:
                acc = torch.sum((y == output).float() * attn)
                total_acc += acc
                total_cnt += torch.sum(attn).item()
            else:
                acc = torch.sum((y == output).float())
                total_acc += acc
                total_cnt += y.shape[0]

            if not seq:
                y_hat = y_hat.permute(0, 2, 1)
            loss = self.compute_loss(y_hat, y, attn, seq)

            if self.weight is not None:
                for param in self.model.parameters():
                    loss += self.weight * torch.norm(param, p=2)

            total_loss += loss

            if mode == 0:
                self.model.zero_grad()
                clip_grad_norm_(self.model.parameters(), 3.0)
                loss.backward()
                self.optim.step()

        if mode == 2:
            return round(total_loss.item() / len(training_data), 4), round(total_acc.item() / total_cnt, 4), all_output
        return round(total_loss.item() / len(training_data), 4), round(total_acc.item() / total_cnt, 4)

    def save_checkpoint(self, epoch, train_acc, valid_acc, valid_loss, train_loss, is_best, filename):
        if is_best:
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
            torch.save(self.model.pianobart.state_dict(), filename[:-5]+"_pianobart.pth")

def main():
    seed = 2024
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = get_args_finetune()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data_finetune(args.dataset, args.dataroot)
    trainset = FinetuneDataset(X=X_train, y=y_train)
    validset = FinetuneDataset(X=X_val, y=y_val)
    testset = FinetuneDataset(X=X_test, y=y_test)

    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print("   len of train_loader", len(train_loader))
    valid_loader = DataLoader(
        validset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader", len(valid_loader))
    test_loader = DataLoader(
        testset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader", len(test_loader))

    print("\nBuilding BART model")
    configuration = BartConfig(max_position_embeddings=args.max_seq_len,
                               d_model=args.hs,
                               encoder_layers=args.layers,
                               encoder_ffn_dim=args.ffn_dims,
                               encoder_attention_heads=args.heads,
                               decoder_layers=args.layers,
                               decoder_ffn_dim=args.ffn_dims,
                               decoder_attention_heads=args.heads
                               )
    pianobart = PianoBart(bartConfig=configuration, e2w=e2w, w2e=w2e)
    if not args.nopretrain:
        best_mdl = args.ckpt
        print("   Loading pre-trained model from", best_mdl.split('/')[-1])
        checkpoint = torch.load(best_mdl, map_location='cpu')
        pianobart.load_state_dict(checkpoint['state_dict'])


    print("\nCreating Finetune Trainer")
    trainer = FinetuneTrainer(pianobart, train_loader, valid_loader, test_loader, args.lr, args.class_num, args.hs, y_test.shape, args.cpu, args.cuda_devices, None, args.seq, args.error_correction, args.weight)

    print("\nTraining Start")
    save_dir = 'result/finetune'
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, args.task+'.ckpt')
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

    with open(os.path.join(save_dir, 'log'), 'a') as outfile:
        for epoch in range(args.epochs):
            train_loss, train_acc = trainer.train()
            valid_loss, valid_acc = trainer.valid()
            test_loss, test_acc, _ = trainer.test()

            is_best = valid_acc >= best_acc
            best_acc = max(valid_acc, best_acc)

            if is_best:
                bad_cnt, best_epoch = 0, epoch
            else:
                bad_cnt += 1

            print(
                'epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {} | Test loss: {} | Test acc: {}'.format(
                    epoch + 1, args.epochs, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc))

            trainer.save_checkpoint(epoch, train_acc, valid_acc,
                                    valid_loss, train_loss, is_best, filename)

            outfile.write(
                'Epoch {}: train_loss={}, valid_loss={}, test_loss={}, train_acc={}, valid_acc={}, test_acc={}\n'.format(
                    epoch + 1, train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc))

            if bad_cnt > 3:
                print('valid acc not improving for 3 epochs')
                break

if __name__ == '__main__':
    main()
