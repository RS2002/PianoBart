import numpy as np

max_window = 1024
max_bar = 255
max_inst = 128
max_pitch = 255
max_velocity = 31
max_pos = 127
max_dura = 127
max_ts = 253
max_tp = 48
tokens_per_note = 8
token_boundary = (max_bar, max_pos, max_inst, max_pitch, 
                    max_dura, max_velocity, max_ts, max_tp)

def data_split(data: np.array):
    m = data.shape[0] // max_window + 1
    pad_num = m * max_window - data.shape[0]
    # padding
    padded = np.append(data, [[i + 1 for i in token_boundary]]*pad_num, axis=0)
    return padded.reshape(m, max_window, tokens_per_note)

default = ('asap', 'EMOPIA', 'Pianist8', 'POP1K7', 'POP909')

for tag in ('train', 'test', 'valid'):
    output = None
    for d in default:
        a = np.load(f'Data\output_pretrain\{d}\{d}_{tag}.npy')
        if output is None:
            output = a
        else:
            output = np.concatenate((output, a), axis=0)
    split = data_split(output)
    np.save(f'Data\output_pretrain\pretrain\pretrain_{tag}.npy', output)
    np.save(f'Data\output_pretrain\pretrain\pretrain_{tag}_split.npy', split)
    print(output.shape, split.shape)

