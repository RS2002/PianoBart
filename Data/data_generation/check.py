import numpy as np

default = ('asap', 'EMOPIA', 'Pianist8', 'POP1K7', 'POP909','')

for flag in ('', '_split'):
    for tag in ('train', 'test', 'valid'):
        for d in default:
            file = f'Data\output\{d}\midi_{tag}{flag}.npy'
            a = np.load(file)
            if flag == '_split':
                m = a[:, :, 0]
                m = m.ravel()
            else:
                m = a[:, 0]
            print(m.shape)
            
            print(f'{file}: {max(m)}')
