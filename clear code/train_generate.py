from transformers import AdamW
import torch
import torch.nn as nn
from model import PianoBartLM
import argparse
import numpy as np
import tqdm
from torch.nn.utils import clip_grad_norm_
import os
import random
from PianoBart import PianoBart
import pickle
from dataset import FinetuneDataset
from torch.utils.data import DataLoader
from transformers import BartConfig


def get_args_generation():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("--datasets", type=str, default='maestro')

    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='./Data/Octuple.pkl')
    parser.add_argument('--name', type=str, default='pianobart')
    parser.add_argument(
        '--ckpt', default='./merge_pianobart.pth')

    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=1024)
    # layer nums of encoder & decoder
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--ffn_dims', type=int, default=2048)  # FFN dims
    parser.add_argument('--heads', type=int, default=8)  # attention heads

    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-6,
                        help='initial learning rate')
    parser.add_argument('--nopretrain', action="store_true")  # default: false

    parser.add_argument('--dataroot', type=str,
                        default=None, help='path to dataset')

    ### cuda ###
    parser.add_argument("--cpu", action="store_true")  # default=False
    parser.add_argument("--cuda_devices", type=int, nargs='+',
                        default=[0], help="CUDA device ids")

    parser.add_argument("--eval", action="store_true")  # default = False

    args = parser.parse_args()

    return args

def load_data(dataset, task, data_root=None):
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
        print('y_train: {}, y_valid: {}, y_test: {}'.format(
            y_train.shape, y_val.shape, y_test.shape))
    else:
        print("Wrong Task")
        exit(0)

    return X_train, X_val, X_test, y_train, y_val, y_test



class GenerationTrainer:
    def __init__(self, pianobart, train_dataloader, valid_dataloader, test_dataloader,
                 lr, testset_shape, cpu, cuda_devices=None, model=None):
        device_name = "cuda"
        if cuda_devices is not None and len(cuda_devices) >= 1:
            device_name += ":" + str(cuda_devices[0])
        self.device = torch.device(
            device_name if torch.cuda.is_available() and not cpu else 'cpu')
        print('   device:', self.device)
        self.pianobart = pianobart
        if model is not None:  # load model
            print('load a fine-tuned model')
            self.model = model.to(self.device)
        else:
            print('init a fine-tune model')
            self.model = PianoBartLM(pianobart).to(self.device)

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
        self.testset_shape = testset_shape

        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def train(self):
        self.model.train()
        torch.set_grad_enabled(True)
        train_loss, train_acc = self.iteration(self.train_data, 0)
        return train_loss, train_acc

    def valid(self):
        self.model.eval()
        torch.set_grad_enabled(False)
        valid_loss, valid_acc = self.iteration(self.valid_data, 1)
        return valid_loss, valid_acc

    def test(self):
        self.model.eval()
        torch.set_grad_enabled(False)
        test_loss, test_acc, all_output = self.iteration(self.test_data, 2)
        return test_loss, test_acc, all_output

    def iteration(self, training_data, mode):
        pbar = tqdm.tqdm(training_data, disable=False)

        total_acc, total_loss = [0] * len(self.pianobart.e2w), 0
        total_cnt = 0

        if mode == 2:  # testing
            all_output = torch.empty(self.testset_shape)
            cnt = 0

        for x, y in pbar:  # (batch, 512, 768)
            batch = x.shape[0]
            x, y = x.to(self.device), y.to(self.device)
            x = x.long()
            y = y.long()
            attn_encoder = (
                x[:, :, 0] != self.pianobart.bar_pad_word).float().to(self.device)
            y_shift = torch.zeros_like(y)
            y_shift[:, 1:, :] = y[:, :-1, :]
            y_shift[:, 0, :] = torch.tensor(self.pianobart.sos_word_np)
            attn_decoder = (
                y_shift[:, :, 0] != self.pianobart.bar_pad_word).float().to(self.device)
            y_hat = self.model.forward(input_ids_encoder=x, input_ids_decoder=y_shift,
                                       encoder_attention_mask=attn_encoder, decoder_attention_mask=attn_decoder)

            outputs = []
            for i, etype in enumerate(self.pianobart.e2w):
                output = np.argmax(y_hat[i].cpu().detach().numpy(), axis=-1)
                outputs.append(output)
            outputs = np.stack(outputs, axis=-1)
            outputs = torch.from_numpy(outputs).to(
                self.device)  # (batch, seq_len)

            if mode == 2:
                all_output[cnt: cnt + batch] = outputs
                cnt += batch

            all_acc = []

            for i in range(8):
                acc = torch.sum(
                    (y[:, :, i] == outputs[:, :, i]).float()*attn_decoder)
                acc /= torch.sum(attn_decoder)
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
                if i == 2 or i == 6 or i == 7:
                    weight = 0.3
                elif i == 3:
                    weight = 1.5
                else:
                    weight = 1
                losses.append(self.compute_loss(
                    y_hat[i], y[..., i], attn_decoder)*weight)
            total_loss_all = [x * y for x, y in zip(losses, n_tok)]
            loss = sum(total_loss_all) / sum(n_tok)  # weighted
            total_loss += loss.item()

            # update only in train
            if mode == 0:
                self.model.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 3.0)  # 用于裁剪梯度，防止梯度爆炸
                self.optim.step()

            accs = list(map(float, all_acc))
            # sys.stdout.write(
            #     'Loss: {:06f} | loss: {:03f}, {:03f}, {:03f}, {:03f}, {:03f}, {:03f}, {:03f}, {:03f}\n'.format(
            #         loss, *losses))
            # sys.stdout.write(
            #     'Acc: {:06f} | acc: {:03f}, {:03f}, {:03f}, {:03f}, {:03f}, {:03f}, {:03f}, {:03f}\n'.format(
            #         np.average(accs), *accs))

        if mode == 2:
            return round(total_loss / len(training_data), 4), [round(x.item() / len(training_data), 4) for x in total_acc],  all_output
        return round(total_loss / len(training_data), 4), [round(x.item() / len(training_data), 4) for x in total_acc]

    def save_checkpoint(self, epoch, train_acc, valid_acc,
                        valid_loss, train_loss, is_best, filename):
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

if __name__ == '__main__':
    # set seed
    seed = 2024
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # current gpu
    torch.cuda.manual_seed_all(seed)  # all gpu
    np.random.seed(seed)
    random.seed(seed)

    # argument
    args = get_args_generation()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("\nLoading Dataset")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(
        dataset=args.datasets, task="gen", data_root=args.dataroot)

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

    best_mdl = args.ckpt
    model = None
    if args.eval:
        print("   Loading pre-trained model from", best_mdl.split('/')[-1])
        checkpoint = torch.load(best_mdl, map_location='cpu')
        model = PianoBartLM(pianobart)
        if best_mdl[-1]=='t':
            model.load_state_dict(checkpoint['state_dict'],strict=False)
        else:
            model.load_state_dict(checkpoint,strict=False)
    elif not args.nopretrain:
        print("   Loading pre-trained model from", best_mdl.split('/')[-1])
        checkpoint = torch.load(best_mdl, map_location='cpu')
        if best_mdl[-1]=='t':
            pianobart.load_state_dict(checkpoint['state_dict'],strict=False)
        else:
            pianobart.load_state_dict(checkpoint,strict=False)

    print("\nCreating Finetune Trainer")
    trainer = GenerationTrainer(pianobart, train_loader, valid_loader, test_loader, args.lr,
                                y_test.shape, args.cpu, args.cuda_devices, model)

    print("\nTraining Start")
    save_dir = os.path.join('result/finetune/generation_' + args.name)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'model.ckpt')
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

    with open(os.path.join(save_dir, 'log'), 'a') as outfile:
        outfile.write("Loading pre-trained model from " +
                      best_mdl.split('/')[-1] + '\n')
        for epoch in range(args.epochs):
            train_loss, train_acc  = trainer.train()
            valid_loss, valid_acc = trainer.valid()
            test_loss, test_acc, _ = trainer.test()

            weighted_score = [
                x * y for (x, y) in zip(valid_acc, pianobart.n_tokens)]
            avg_acc = sum(weighted_score) / sum(pianobart.n_tokens)

            is_best = avg_acc > best_acc
            best_acc = max(avg_acc, best_acc)

            if is_best:
                bad_cnt, best_epoch = 0, epoch
            else:
                bad_cnt += 1

            print(
                'epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {} | Test loss: {} | Test acc: {} '.format(
                    epoch + 1, args.epochs, train_loss, train_acc, valid_loss, valid_acc,  test_loss, test_acc))

            trainer.save_checkpoint(epoch, train_acc, valid_acc,
                                    valid_loss, train_loss, is_best, filename)

            outfile.write(
                'Epoch {}: train_loss={}, valid_loss={}, test_loss={}, train_acc={}, valid_acc={}, test_acc={}\n'.format(
                    epoch + 1, train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc))

            if bad_cnt > 30:
                print('valid acc not improving for 30 epochs')
                break