import argparse
import numpy as np
import random
import pickle
import os
import json
from torch.utils.data import DataLoader
from transformers import BartConfig
from PianoBart import PianoBart
from pretrain import Pretrainer
from model import MidiDataset


def get_args():
    parser = argparse.ArgumentParser(description='')

    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='./Data/Octuple.pkl')
    parser.add_argument('--name', type=str, default='PianoBart')

    ### pre-train dataset ###
    parser.add_argument("--datasets", type=str, nargs='+', default=['asap', 'EMOPIA', 'Pianist8', 'POP1K7', 'POP909']) #TODO

    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--mask_percent', type=float, default=0.15,
                        help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction")
    parser.add_argument('--max_seq_len', type=int, default=1024, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=1536)  # hidden state
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')

    ### cuda ###
    parser.add_argument("--cpu", action="store_true")  # default: False
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7], help="CUDA device ids")

    args = parser.parse_args()

    return args


def load_data(datasets,mode):
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

def pretrain():
    args = get_args()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("\nLoading Dataset", args.datasets)
    X_train, X_val = load_data(datasets=args.datasets,mode="pretrain")

    trainset = MidiDataset(X=X_train)
    validset = MidiDataset(X=X_val)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print("   len of train_loader", len(train_loader))
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader", len(valid_loader))

    print("\nBuilding BART model")
    configuration = BartConfig(max_position_embeddings=args.max_seq_len,
                               d_model=args.hs)

    pianobart = PianoBart(bartConfig=configuration, e2w=e2w, w2e=w2e)
    print("\nCreating BART Trainer")
    trainer = Pretrainer(pianobart, train_loader, valid_loader, args.lr, args.batch_size, args.max_seq_len,
                          args.mask_percent, args.cpu, args.cuda_devices)

    print("\nTraining Start")
    save_dir = 'PianoBart/result/pretrain/' + args.name
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'model.ckpt')
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

    for epoch in range(args.epochs):
        if bad_cnt >= 30:
            print('valid acc not improving for 30 epochs')
            break
        train_loss, train_acc = trainer.train()
        valid_loss, valid_acc = trainer.valid()

        weighted_score = [x * y for (x, y) in zip(valid_acc, pianobart.n_tokens)]
        avg_acc = sum(weighted_score) / sum(pianobart.n_tokens)

        is_best = avg_acc > best_acc
        best_acc = max(avg_acc, best_acc)

        if is_best:
            bad_cnt, best_epoch = 0, epoch
        else:
            bad_cnt += 1

        print('epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {}'.format(
            epoch + 1, args.epochs, train_loss, train_acc, valid_loss, valid_acc))

        trainer.save_checkpoint(epoch, best_acc, valid_acc,
                                valid_loss, train_loss, is_best, filename)

        with open(os.path.join(save_dir, 'log'), 'a') as outfile:
            outfile.write('Epoch {}: train_loss={}, train_acc={}, valid_loss={}, valid_acc={}\n'.format(
                epoch + 1, train_loss, train_acc, valid_loss, valid_acc))


if __name__ == '__main__':
    pretrain()