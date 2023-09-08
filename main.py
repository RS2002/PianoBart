import os
import numpy as np
import random
import pickle
import time
from torch.utils.data import DataLoader
from transformers import BartConfig
from PianoBart import PianoBart
from pretrain import Pretrainer,get_args_pretrain,load_data_pretrain
from finetune import FinetuneTrainer,get_args_finetune,load_data_finetune
import torch
from dataset import MidiDataset,FinetuneDataset
from model import TokenClassification,SequenceClassification,PianoBartLM
from eval import get_args_eval,load_data_eval,conf_mat
from finetune_generation import get_args_generation,GenerationTrainer
from eval_generation import get_args_eval_generation
# import json
from Ablation import AblationTrainer, load_data_ablation, get_args_ablation

def pretrain():
    args = get_args_pretrain()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("\nLoading Dataset", args.datasets)
    X_train, X_val = load_data_pretrain(datasets=args.datasets,mode="pretrain")

    trainset = MidiDataset(X=X_train)
    validset = MidiDataset(X=X_val)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print("   len of train_loader", len(train_loader))
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader", len(valid_loader))

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
    print("\nCreating BART Trainer")
    trainer = Pretrainer(pianobart, train_loader, valid_loader, args.lr, args.batch_size, args.max_seq_len,
                          args.mask_percent, args.cpu, args.cuda_devices)

    print("\nTraining Start")
    save_dir = 'result/pretrain/' + args.name
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'model.ckpt')
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

    start_t = time.time()

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
    
    end_t = time.time()

    print(f'Time cost in pretrain of PianoBart is {end_t - start_t}, start_t = {start_t}, end_t = {end_t}')
    with open(os.path.join(save_dir, 'log'), 'a') as outfile:
            outfile.write(f'Time cost in pretrain of PianoBart is {end_t - start_t}, start_t = {start_t}, end_t = {end_t}')

def finetune():
    # set seed
    seed = 2023
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # current gpu
    torch.cuda.manual_seed_all(seed)  # all gpu
    np.random.seed(seed)
    random.seed(seed)

    # argument
    args = get_args_finetune()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("\nLoading Dataset")
    if args.task == 'melody' or args.task == 'velocity':
        dataset = 'pop909'
        seq_class = False
    elif args.task == 'composer':
        dataset = 'composer'
        seq_class = True
    elif args.task == 'emotion':
        dataset = 'emopia'
        seq_class = True
    else:
        print("ERROR")
        exit(-1)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data_finetune(args.dataset, args.task, args.dataroot)

    trainset = FinetuneDataset(X=X_train, y=y_train)
    validset = FinetuneDataset(X=X_val, y=y_val)
    testset = FinetuneDataset(X=X_test, y=y_test)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print("   len of train_loader", len(train_loader))
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader", len(valid_loader))
    test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
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

    best_mdl = ''
    if not args.nopretrain:
        best_mdl = args.ckpt
        print("   Loading pre-trained model from", best_mdl.split('/')[-1])
        checkpoint = torch.load(best_mdl, map_location='cpu')
        pianobart.load_state_dict(checkpoint['state_dict'])

    print("\nCreating Finetune Trainer")
    trainer = FinetuneTrainer(pianobart, train_loader, valid_loader, test_loader, args.lr, args.class_num,
                              args.hs, y_test.shape, args.cpu, args.cuda_devices, None, seq_class)

    print("\nTraining Start")
    save_dir = os.path.join('result/finetune/', args.task + '_' + args.name)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'model.ckpt')
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

    #    train_accs, valid_accs = [], []
    with open(os.path.join(save_dir, 'log'), 'a') as outfile:
        outfile.write("Loading pre-trained model from " + best_mdl.split('/')[-1] + '\n')
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

            #            train_accs.append(train_acc)
            #            valid_accs.append(valid_acc)
            trainer.save_checkpoint(epoch, train_acc, valid_acc,
                                    valid_loss, train_loss, is_best, filename)

            outfile.write(
                'Epoch {}: train_loss={}, valid_loss={}, test_loss={}, train_acc={}, valid_acc={}, test_acc={}\n'.format(
                    epoch + 1, train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc))

            if bad_cnt > 3:
                print('valid acc not improving for 3 epochs')
                break

    # draw figure valid_acc & train_acc
    '''plt.figure()
    plt.plot(train_accs)
    plt.plot(valid_accs)
    plt.title(f'{args.task} task accuracy (w/o pre-training)')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train','valid'], loc='upper left')
    plt.savefig(f'acc_{args.task}_scratch.jpg')'''


def eval():
    args = get_args_eval()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

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

    print("\nLoading Dataset")
    if args.task == 'melody' or args.task == 'velocity':
        dataset = 'pop909'
        model = TokenClassification(pianobart, args.class_num, args.hs)
        seq_class = False
    elif args.task == 'composer' or args.task == 'emotion':
        dataset = args.task
        model = SequenceClassification(pianobart, args.class_num, args.hs)
        seq_class = True
    else:
        print("ERROR")
        exit(-1)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data_eval(dataset, args.task)

    '''trainset = FinetuneDataset(X=X_train, y=y_train)
    validset = FinetuneDataset(X=X_val, y=y_val)'''
    testset = FinetuneDataset(X=X_test, y=y_test)

    '''train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print("   len of train_loader", len(train_loader))
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader", len(valid_loader))'''
    train_loader,valid_loader=None,None

    test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of test_loader", len(test_loader))

    print('\nLoad ckpt from', args.ckpt)
    best_mdl = args.ckpt
    checkpoint = torch.load(best_mdl, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    # remove module
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint['state_dict'].items():
    #    name = k[7:]
    #    new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)

    print("\nCreating Finetune Trainer")
    trainer = FinetuneTrainer(pianobart, train_loader, valid_loader, test_loader, args.lr, args.class_num,
                              args.hs, y_test.shape, args.cpu, args.cuda_devices, model, seq_class)

    test_loss, test_acc, all_output = trainer.test()
    print('test loss: {}, test_acc: {}'.format(test_loss, test_acc))

    outdir = os.path.dirname(args.ckpt)
    conf_mat(y_test, all_output, args.task, outdir)

def finetune_generation():
    # set seed
    seed = 2023
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
    #X_train, X_val, X_test, y_train, y_val, y_test = load_data_finetuneload_data_finetune(args.dataset, args.task, args.dataroot)
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_finetune(dataset=args.datasets,task="gen",data_root="./Data/output_generation/maestro/gen_method")

    trainset = FinetuneDataset(X=X_train, y=y_train)
    validset = FinetuneDataset(X=X_val, y=y_val)
    testset = FinetuneDataset(X=X_test, y=y_test)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print("   len of train_loader", len(train_loader))
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader", len(valid_loader))
    test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
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

    best_mdl = ''
    if not args.nopretrain:
        best_mdl = args.ckpt
        print("   Loading pre-trained model from", best_mdl.split('/')[-1])
        checkpoint = torch.load(best_mdl, map_location='cpu')
        pianobart.load_state_dict(checkpoint['state_dict'])

    print("\nCreating Finetune Trainer")
    trainer = GenerationTrainer(pianobart, train_loader, valid_loader, test_loader, args.lr,
                               y_test.shape, args.cpu, args.cuda_devices, None)

    print("\nTraining Start")
    save_dir = os.path.join('result/finetune/generation_' + args.name)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'model.ckpt')
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

    #    train_accs, valid_accs = [], []
    with open(os.path.join(save_dir, 'log'), 'a') as outfile:
        outfile.write("Loading pre-trained model from " + best_mdl.split('/')[-1] + '\n')
        for epoch in range(args.epochs):
            train_loss, train_acc, train_FAD_BAR, train_FAD,  = trainer.train()
            valid_loss, valid_acc, valid_FAD_BAR, valid_FAD = trainer.valid()
            test_loss, test_acc, test_FAD_BAR, test_FAD, _ = trainer.test()

            is_best = np.mean(valid_acc) >= np.mean(best_acc)
            best_acc = max(np.mean(valid_acc), np.mean(best_acc))

            if is_best:
                bad_cnt, best_epoch = 0, epoch
            else:
                bad_cnt += 1

            print(
                'epoch: {}/{} | Train Loss: {} | Train acc: {} | Train FAD: {} | Train FAD (BAR): {} | Valid Loss: {} | Valid acc: {} | Valid FAD: {} | Valid FAD(BAR): {} | Test loss: {} | Test acc: {} | Test FAD: {} | Test FAD(BAR): {}'.format(
                    epoch + 1, args.epochs, train_loss, train_acc, train_FAD, train_FAD_BAR, valid_loss, valid_acc, valid_FAD, valid_FAD_BAR, test_loss, test_acc, test_FAD, test_FAD_BAR))

            #            train_accs.append(train_acc)
            #            valid_accs.append(valid_acc)
            trainer.save_checkpoint(epoch, train_acc, valid_acc,
                                    valid_loss, train_loss, is_best, filename)

            outfile.write(
                'Epoch {}: train_loss={}, valid_loss={}, test_loss={}, train_acc={}, valid_acc={}, test_acc={}, train_fad={}, valid_fad={}, test_fad={}, train_fad(bar)={}, valid_fad(bar)={}, test_fad(bar)={}\n'.format(
                    epoch + 1, train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc, train_FAD, valid_FAD, test_FAD, train_FAD_BAR, valid_FAD_BAR, test_FAD_BAR))

            if bad_cnt > 3:
                print('valid acc not improving for 3 epochs')
                break

    # draw figure valid_acc & train_acc
    '''plt.figure()
    plt.plot(train_accs)
    plt.plot(valid_accs)
    plt.title(f'{args.task} task accuracy (w/o pre-training)')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train','valid'], loc='upper left')
    plt.savefig(f'acc_{args.task}_scratch.jpg')'''


def eval_generation():
    args = get_args_eval_generation()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

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
    model = PianoBartLM(pianobart)

    print("\nLoading Dataset")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data_finetune(args.dataset, "gen")

    '''trainset = FinetuneDataset(X=X_train, y=y_train)
    validset = FinetuneDataset(X=X_val, y=y_val)'''
    testset = FinetuneDataset(X=X_test, y=y_test)

    '''train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print("   len of train_loader", len(train_loader))
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader", len(valid_loader))'''
    train_loader,valid_loader=None,None

    test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of test_loader", len(test_loader))

    print('\nLoad ckpt from', args.ckpt)
    best_mdl = args.ckpt
    checkpoint = torch.load(best_mdl, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    # remove module
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint['state_dict'].items():
    #    name = k[7:]
    #    new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)

    print("\nCreating Finetune Trainer")
    trainer = GenerationTrainer(pianobart, train_loader, valid_loader, test_loader, args.lr,
                               y_test.shape, args.cpu, args.cuda_devices, model)

    test_loss, test_acc, all_output = trainer.test()
    print('test loss: {}, test_acc: {}'.format(test_loss, test_acc))

    outdir = os.path.dirname(args.ckpt)
    conf_mat(y_test, all_output, args.task, outdir)

def abalation():
    # set seed
    seed = 2023
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
    X_train, X_val, X_test = load_data_ablation(datasets=args.datasets, mode="ablation")

    trainset = MidiDataset(X=X_train)
    validset = MidiDataset(X=X_val)
    testset = MidiDataset(X=X_test)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print("   len of train_loader", len(train_loader))
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader", len(valid_loader))
    test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
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

    best_mdl = ''
    if not args.nopretrain:
        best_mdl = args.ckpt
        print("   Loading pre-trained model from", best_mdl.split('/')[-1])
        checkpoint = torch.load(best_mdl, map_location='cpu')
        pianobart.load_state_dict(checkpoint['state_dict'])

    print("\nCreating Finetune Trainer")
    trainer = AblationTrainer(pianobart, train_loader, valid_loader, test_loader, args.lr, X_test.shape, args.cpu, args.cuda_devices, None)

    print("\nTraining Start")
    save_dir = os.path.join('result/finetune/generation_' + args.name)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'model.ckpt')
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

    #    train_accs, valid_accs = [], []
    with open(os.path.join(save_dir, 'log'), 'a') as outfile:
        outfile.write("Loading pre-trained model from " + best_mdl.split('/')[-1] + '\n')
        for epoch in range(args.epochs):
            train_loss, train_acc, train_FAD_BAR, train_FAD,  = trainer.train()
            valid_loss, valid_acc, valid_FAD_BAR, valid_FAD = trainer.valid()
            test_loss, test_acc, test_FAD_BAR, test_FAD, _ = trainer.test()

            is_best = np.mean(valid_acc) >= np.mean(best_acc)
            best_acc = max(np.mean(valid_acc), np.mean(best_acc))

            if is_best:
                bad_cnt, best_epoch = 0, epoch
            else:
                bad_cnt += 1

            print(
                'epoch: {}/{} | Train Loss: {} | Train acc: {} | Train FAD: {} | Train FAD (BAR): {} | Valid Loss: {} | Valid acc: {} | Valid FAD: {} | Valid FAD(BAR): {} | Test loss: {} | Test acc: {} | Test FAD: {} | Test FAD(BAR): {}'.format(
                    epoch + 1, args.epochs, train_loss, train_acc, train_FAD, train_FAD_BAR, valid_loss, valid_acc, valid_FAD, valid_FAD_BAR, test_loss, test_acc, test_FAD, test_FAD_BAR))

            #            train_accs.append(train_acc)
            #            valid_accs.append(valid_acc)
            trainer.save_checkpoint(epoch, train_acc, valid_acc,
                                    valid_loss, train_loss, is_best, filename)

            outfile.write(
                'Epoch {}: train_loss={}, valid_loss={}, test_loss={}, train_acc={}, valid_acc={}, test_acc={}, train_fad={}, valid_fad={}, test_fad={}, train_fad(bar)={}, valid_fad(bar)={}, test_fad(bar)={}\n'.format(
                    epoch + 1, train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc, train_FAD, valid_FAD, test_FAD, train_FAD_BAR, valid_FAD_BAR, test_FAD_BAR))


            if bad_cnt > 3:
                print('valid acc not improving for 3 epochs')
                break




'''
to run finetune, for example if use Piani8 dataset:
python main.py --task composer --dataset Pianist8 --class_num --dataroot ./Data/output_composer/Pianist8 --cuda_devices 0
'''
if __name__ == '__main__':
    #pretrain()
    #finetune()
    #eval()
    finetune_generation()
    #finetune_eval()
    #abalation()
