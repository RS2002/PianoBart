import argparse


def get_args_eval_generation():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("--datasets", type=str, nargs='+', default=[]) #TODO

    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='../../Data/Octuple.pkl')
    parser.add_argument('--ckpt', type=str, default='')

    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--mask_percent', type=float, default=0.15,
                        help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction")
    parser.add_argument('--max_seq_len', type=int, default=1024, help='all sequences are padded to `max_seq_len`') #TODO:512
    parser.add_argument('--hs', type=int, default=768)  # hidden state
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')

    ### cuda ###
    parser.add_argument('--cpu', action="store_true")  # default: false
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0, 1], help="CUDA device ids")

    args = parser.parse_args()

    root = 'result/finetune/'
    args.ckpt = root + 'generation_midibert/model_best.ckpt' if args.ckpt == '' else args.ckpt


    return args
