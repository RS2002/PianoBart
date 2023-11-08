import argparse
import pickle
from transformers import BartConfig
from PianoBart import PianoBart
from model import PianoBartLM
import torch
import torch.nn as nn
import numpy as np
import os
from dataset import MidiDataset
import tqdm
from torch.utils.data import DataLoader



def get_args():
    parser = argparse.ArgumentParser(description='')

    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='./Data/Octuple.pkl')
    parser.add_argument('--ckpt', type=str, default='result/pretrain/pianobart/model_best.ckpt')
    parser.add_argument('--dataset_path', type=str, default='./Data/output_generate/GiantMIDI1k/gen_method')
    parser.add_argument('--dataset_name', type=str, default='GiantMIDI1k_test.npy')
    parser.add_argument('--output', type=str, default='./output.npy')

    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_seq_len', type=int, default=1024, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=1024)
    parser.add_argument('--layers', type=int, default=8)  # layer nums of encoder & decoder
    parser.add_argument('--ffn_dims', type=int, default=2048)  # FFN dims
    parser.add_argument('--heads', type=int, default=8)  # attention heads

    parser.add_argument('--nopretrain', action="store_true",default=False)  # default: false

    ### cuda ###
    parser.add_argument("--cpu", action="store_true")  # default=False
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0], help="CUDA device ids")

    args = parser.parse_args()
    return args

def load_data(dataset_path,dataset_name):
    data_path=os.path.join(dataset_path,dataset_name)
    dataset=np.load(data_path,allow_pickle=True)
    return dataset

if __name__ == '__main__':
    args = get_args()

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
    if not args.nopretrain:
        best_mdl = args.ckpt
        print("   Loading pre-trained model from", best_mdl.split('/')[-1])
        checkpoint = torch.load(best_mdl, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'],strict=False)

    print("\nLoading Dataset", args.dataset_name)
    data=load_data(args.dataset_path,args.dataset_name)
    dataset=MidiDataset(data)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    num=len(data_loader)
    print("   len of dataset", num)

    device_name = "cuda"
    cpu = args.cpu
    cuda_devices = args.cuda_devices
    if cuda_devices is not None and len(cuda_devices) >= 1:
        device_name += ":" + str(cuda_devices[0])
    device = torch.device(device_name if torch.cuda.is_available() and not cpu else 'cpu')
    if len(cuda_devices) > 1 and not cpu:
        print("Use %d GPUS" % len(cuda_devices))
        model = nn.DataParallel(model, device_ids=cuda_devices)
    elif (len(cuda_devices) == 1 or torch.cuda.is_available()) and not cpu:
        print("Use GPU", end=" ")
        print(device)
        model = model.to(device)
    else:
        print("Use CPU")

    print("\nEval Start")
    pabr=tqdm.tqdm(data_loader, disable=False)

    output=torch.zeros([num,args.max_seq_len,8])
    cnt=0
    for x in pabr:
        x=x.long().to(device)
        batch=x.shape[0]
        attn_encoder = (x[:, :, 0] != pianobart.bar_pad_word).float().to(device)

        if len(args.cuda_devices) == 0:
            device_num = -1
        else:
            device_num = args.cuda_devices[0]

        y = model(input_ids_encoder=x, encoder_attention_mask=attn_encoder, generate=True, device_num=device_num)
        output[cnt:cnt+batch,...]=y
        cnt+=batch
    np.save(args.output,output.detach().cpu().numpy())
