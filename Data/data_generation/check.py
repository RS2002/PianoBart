import numpy as np
# 0 Measure, 1 Pos, 2 Program, 3 Pitch, 4 Duration, 5 Velocity, 6 TimeSig, 7 Tempo
default = ('asap', 'EMOPIA', 'Pianist8', 'POP1K7', 'POP909','')
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
for flag in ('', '_split'):
    for tag in ('train', 'test', 'valid'):
        for d in default:
            file = f'Data\output\{d}\midi_{tag}{flag}.npy'
            a = np.load(file)
            for num in range(8):
                if flag == '_split':
                    m = a[:, :, num]
                    m = m.ravel()
                else:
                    m = a[:, num]
                print(f'{file}|{maps[num]}: {max(m)==mmaps[num]}')
