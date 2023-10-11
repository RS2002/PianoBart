import argparse
import pickle
from transformers import BartConfig
from PianoBart import PianoBart
from model import PianoBartLM
import torch
import torch.nn as nn
import numpy as np
import miditoolkit
from Data.data_generation.convert import MIDI_to_encoding, encoding_to_MIDI, padding

def get_args():
    parser = argparse.ArgumentParser(description='')

    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='./Data/Octuple.pkl')
    parser.add_argument('--ckpt', default='result/pretrain/pianobart/model_best.ckpt')
    parser.add_argument('--input',default='./Data/POP909/POP909/001/001.mid')
    parser.add_argument('--output',default='./output.mid')

    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--max_seq_len', type=int, default=1024, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=1024)
    parser.add_argument('--layers', type=int, default=8)  # layer nums of encoder & decoder
    parser.add_argument('--ffn_dims', type=int, default=2048)  # FFN dims
    parser.add_argument('--heads', type=int, default=8)  # attention heads

    parser.add_argument('--nopretrain', action="store_true",default=False)  # default: false

    ### cuda ###
    parser.add_argument("--cpu", action="store_true")  # default=False
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[5, 6, 7], help="CUDA device ids")

    args = parser.parse_args()

    return args


def Midi2Octuple(Midi_path):
    midi_obj = miditoolkit.midi.parser.MidiFile(Midi_path)
    encoding = MIDI_to_encoding(midi_obj, task='pretrain')
    encoding = padding(file_name=Midi_path, e_segment=encoding, window=1024)
    encoding = torch.Tensor([encoding]).to(torch.int)
    return encoding


def Octuple2Midi(octuple, Midi_path):
    octuple=torch.squeeze(octuple)
    end_flag=False
    pad=torch.tensor([256,128,129,256,128,32,254,49])
    eos=pad+3

    for i in range(1024):
        for j in range(8):
            if octuple[i,j]>=pad[j] or (j==0 and octuple[i,j]>127): #暂时不生成鼓
                end_flag=True
                octuple[i]=eos
                octuple[i+1:]=pad
                break
        if end_flag:
            break

    if not end_flag:
        octuple[-1]=eos

    octuple = octuple.tolist()
    f = 0
    for i, line in enumerate(octuple):
        if line[0] == 259:
            octuple = octuple[f:i]
            break

    midi = encoding_to_MIDI(octuple)
    midi.dump(Midi_path)



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
        model.load_state_dict(checkpoint['state_dict'])
    input=Midi2Octuple(args.input)
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
        model=model.to(device)
    else:
        print("Use CPU")
    input=input.to(device)
    input = input.long()
    attn_encoder = (input[:, :, 0] != pianobart.bar_pad_word).float().to(device)

    y=model(input_ids_encoder=input,encoder_attention_mask=attn_encoder,generate=True)

    outputs = []
    for i, etype in enumerate(pianobart.e2w):
        output = np.argmax(y[i].cpu().detach().numpy(), axis=-1)
        outputs.append(output)
    outputs = np.stack(outputs, axis=-1)
    outputs = torch.from_numpy(outputs)
    Octuple2Midi(outputs,args.output)
