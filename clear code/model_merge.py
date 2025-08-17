from model_merging_methods.merging_methods import MergingMethod
import argparse
from PianoBart import PianoBart
import pickle
from transformers import BartConfig
import torch

def get_args_finetune():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dict_file', type=str, default='./Data/Octuple.pkl')

    parser.add_argument('--pretrain_ckpt', type=str, default='./pianobart.ckpt')
    parser.add_argument('--model_path',  type=str, default='') # split by ','
    parser.add_argument('--output_path', type=str, default='./merge_pianobart.pth')

    parser.add_argument('--max_seq_len', type=int, default=1024, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=1024)
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--ffn_dims', type=int, default=2048)
    parser.add_argument('--heads', type=int, default=8)

    parser.add_argument('--mask_apply_method', type=str, default="average_merging")
    parser.add_argument('--mask_strategy', type=str, default="random")
    parser.add_argument('--use_weight_rescale', action='store_false') #默认为True
    parser.add_argument('--weight_format', type=str, default="delta_weight")
    parser.add_argument('--weight_mask_rate', type=float, default=0.8)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args_finetune()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

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
    checkpoint = torch.load(args.pretrain_ckpt, map_location='cpu')
    pianobart.load_state_dict(checkpoint['state_dict'],strict=False)

    model_list=args.model_path
    if len(model_list)<=1:
        print("No Moudle to Merge!")
        exit(0)
    finetune_models=[]

    model_list=model_list.split(",")
    print(model_list)
    for model_path in model_list:
        print("Load From "+model_path)
        model=PianoBart(bartConfig=configuration, e2w=e2w, w2e=w2e)
        model.load_state_dict(torch.load(model_path),strict=False)
        finetune_models.append(model)


    weight_mask_rate = [args.weight_mask_rate]*len(finetune_models)
    merging_method = MergingMethod(merging_method_name="mask_merging")
    merged_model = merging_method.get_merged_model(merged_model=pianobart,
                                                   models_to_merge=finetune_models,
                                                   exclude_param_names_regex=[],
                                                   weight_format=args.weight_format,
                                                   weight_mask_rates=weight_mask_rate,
                                                   use_weight_rescale=args.use_weight_rescale,
                                                   mask_strategy=args.mask_strategy,
                                                   mask_apply_method=args.mask_apply_method,
                                                   models_use_deepcopy=True)
    torch.save(merged_model.state_dict(),args.output_path)