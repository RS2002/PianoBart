import numpy as np

default = ('asap', 'EMOPIA', 'Pianist8', 'POP1K7', 'POP909')

for flag in ('', '_split'):
    for tag in ('train', 'test', 'valid'):
        output = None
        for d in default:
            a = np.load(f'Data\output\{d}\midi_{tag}{flag}.npy')
            if output is None:
                output = a
            else:
                output = np.concatenate((output, a), axis=0)
        np.save(f'Data\output\midi_{tag}{flag}.npy', output)
        print(output.shape)
