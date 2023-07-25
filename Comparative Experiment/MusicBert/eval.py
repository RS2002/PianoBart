"""
Evaluate the model on fine-tuning task (melody, velocity, composer, emotion)
Return loss, accuracy, confusion matrix.
"""
import argparse
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools




def get_args_eval():
    parser = argparse.ArgumentParser(description='')

    ### mode ###
    parser.add_argument('--task', choices=['melody', 'velocity', 'composer', 'emotion'], required=True)

    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='../../Data/Octuple.pkl')
    parser.add_argument('--ckpt', type=str, default='')

    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--class_num', type=int)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--max_seq_len', type=int, default=1024, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=1024)
    parser.add_argument('--layers', type=int, default=8)  # layer nums of encoder & decoder
    parser.add_argument('--ffn_dims', type=int, default=2048)  # FFN dims
    parser.add_argument('--heads', type=int, default=8)  # attention heads
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')

    ### cuda ###
    parser.add_argument('--cpu', action="store_true")  # default: false
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0, 1, 2, 3], help="CUDA device ids")

    args = parser.parse_args()

    root = 'result/finetune/'

    if args.task == 'melody':
        args.class_num = 4
        args.ckpt = root + 'melody_musicbert/model_best.ckpt' if args.ckpt == '' else args.ckpt
    elif args.task == 'velocity':
        args.class_num = 7
        args.ckpt = root + 'velocity_musicbert/model_best.ckpt' if args.ckpt == '' else args.ckpt
    elif args.task == 'composer':
        args.class_num = 8
        args.ckpt = root + 'composer_musicbert/model_best.ckpt' if args.ckpt == '' else args.ckpt
    elif args.task == 'emotion':
        args.class_num = 4
        args.ckpt = root + 'emotion_musicbert/model_best.ckpt' if args.ckpt == '' else args.ckpt

    return args


def load_data_eval(dataset, task):
    data_root = '../../Data/finetune/others'

    if dataset == 'emotion':
        dataset = 'emopia'

    if dataset not in ['pop909', 'composer', 'emopia']:
        print('dataset {} not supported'.format(dataset))
        exit(1)

    X_train = np.load(os.path.join(data_root, f'{dataset}_train.npy'), allow_pickle=True)
    X_val = np.load(os.path.join(data_root, f'{dataset}_valid.npy'), allow_pickle=True)
    X_test = np.load(os.path.join(data_root, f'{dataset}_test.npy'), allow_pickle=True)

    print('X_train: {}, X_valid: {}, X_test: {}'.format(X_train.shape, X_val.shape, X_test.shape))

    if dataset == 'pop909':
        y_train = np.load(os.path.join(data_root, f'{dataset}_train_{task[:3]}ans.npy'), allow_pickle=True)
        y_val = np.load(os.path.join(data_root, f'{dataset}_valid_{task[:3]}ans.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(data_root, f'{dataset}_test_{task[:3]}ans.npy'), allow_pickle=True)
    else:
        y_train = np.load(os.path.join(data_root, f'{dataset}_train_ans.npy'), allow_pickle=True)
        y_val = np.load(os.path.join(data_root, f'{dataset}_valid_ans.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(data_root, f'{dataset}_test_ans.npy'), allow_pickle=True)

    print('y_train: {}, y_valid: {}, y_test: {}'.format(y_train.shape, y_val.shape, y_test.shape))

    return X_train, X_val, X_test, y_train, y_val, y_test


def conf_mat(_y, output, task, outdir):
    if task == 'melody':
        target_names = ['M', 'B', 'A']
        seq = False
    elif task == 'velocity':
        target_names = ['pp', 'p', 'mp', 'mf', 'f', 'ff']
        seq = False
    elif task == 'composer':
        target_names = ['M', 'C', 'E', 'H', 'W', 'J', 'S', 'Y']
        seq = True
    elif task == 'emotion':
        target_names = ['HAHV', 'HALV', 'LALV', 'LAHV']
        seq = True

    output = output.detach().cpu().numpy()
    output = output.reshape(-1, 1)
    _y = _y.reshape(-1, 1)

    cm = confusion_matrix(_y, output)

    _title = 'BART (OCTUPLE): ' + task + ' task'

    save_cm_fig(cm, classes=target_names, normalize=False,
                title=_title, outdir=outdir, seq=seq)


def save_cm_fig(cm, classes, normalize, title, outdir, seq):
    if not seq:
        cm = cm[1:, 1:]  # exclude padding

    if normalize:
        cm = cm.astype('float') * 100 / cm.sum(axis=1)[:, None]

    #    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=15,
                 horizontalalignment='center',
                 color='white' if cm[i, j] > threshold else 'black')
    plt.xlabel('predicted', fontsize=18)
    plt.ylabel('true', fontsize=18)
    plt.tight_layout()

    plt.savefig(f'{outdir}/cm_{title.split()[2]}.jpg')
    return


