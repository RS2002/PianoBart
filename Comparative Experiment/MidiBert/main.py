import os
import pickle
from torch.utils.data import DataLoader
from transformers import BertConfig
from MidiBert import MidiBert
from pretrain import Pretrainer,get_args_pretrain,load_data_pretrain
from dataset import MidiDataset


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

    print("\nBuilding BERT model")
    configuration = BertConfig(max_position_embeddings=args.max_seq_len,
                               position_embedding_type='relative_key_query',
                               hidden_size=args.hs)
    midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)
    print("\nCreating BERT Trainer")
    trainer = Pretrainer(midibert, train_loader, valid_loader, args.lr, args.batch_size, args.max_seq_len,
                          args.mask_percent, args.cpu, args.cuda_devices)

    print("\nTraining Start")
    save_dir = '/result/pretrain/' + args.name
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

        n_tokens = [midibert.n_tokens[i] for i in [0,1,3,4]]
        weighted_score = [x * y for (x, y) in zip(valid_acc, n_tokens)]
        avg_acc = sum(weighted_score) / sum(n_tokens)

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
    #finetune()
    #eval()
    #finetune_generation()
    #finetune_eval()