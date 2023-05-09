import numpy as np

default = ('asap', 'EMOPIA', 'Pianist8', 'POP1K7', 'POP909')

for tag in ('train', 'test', 'valid'):
    output = None
    for d in default:
        # a = np.load(f'Data\output\{d}\midi_{tag}.npy')
        a = np.load(f'Data\output\{d}\midi_{tag}_split.npy')
        if output is None:
            output = a
        else:
            output = np.concatenate((output, a), axis=0)
    # np.save(f'Data\output\midi_{tag}.npy', output)
    np.save(f'Data\output\midi_{tag}_split.npy', output)
    print(output.shape)
