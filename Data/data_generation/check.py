import numpy as np
# 0 Measure, 1 Pos, 2 Program, 3 Pitch, 4 Duration, 5 Velocity, 6 TimeSig, 7 Tempo
default = ('asap', 'EMOPIA', 'Pianist8', 'POP1K7', 'POP909','')
composer = ('Pianist8', 'asap')

datasets = {'composer': composer}

maps = {
    0: 'bar',
    1: 'instrument',
    2: 'program',
    3: 'pitch',
    4: 'duration',
    5: 'velocity',
    6: 'timesig',
    7: 'tempo'
}
mmaps = {
    0: 259,
    1: 131,
    2: 132,
    3: 259,
    4: 131,
    5: 35,
    6: 257,
    7: 52
}
def checkPretrain():
    for flag in ('', '_split'):
        for tag in ('train', 'test', 'valid'):
            for d in default:
                file = f'Data\output\{d}\midi_{tag}{flag}.npy'
                a = np.load(file)
                print(a.shape)
                for num in range(8):
                    if flag == '_split':
                        m = a[:, :, num]
                        m = m.ravel()
                    else:
                        m = a[:, num]
                        print(f'{file}|{maps[num]}: {max(m)==mmaps[num]}')

def checkFintune(task):
    for tag in ('train', 'test', 'valid'):
        for d in datasets[task]:
            print(tag, d)
            file = f'Data\output_{task}\{d}\{d}_{tag}.npy'
            ans  = f'Data\output_{task}\{d}\{d}_{tag}_{task[:3]}ans.npy'
            a = np.load(file)
            b = np.load(ans)
            print(a.shape, b.shape, np.min(b), np.max(b))
            for num in range(8):
                m = a[:, :, num]
                m = m.ravel()
                print(f'{file}|{maps[num]}: {max(m)==mmaps[num]}')

checkFintune('composer')