import numpy as np
import random
from midi_to_octuple import encoding_to_MIDI
import os
# 0 Measure, 1 Pos, 2 Program, 3 Pitch, 4 Duration, 5 Velocity, 6 TimeSig, 7 Tempo
default = ( 'EMOPIA', 'Pianist8', 'POP1K7', 'POP909')
composer = ('Pianist8', 'asap')
generate = ('maestro',)
melody = ('POP909',)
datasets = {'composer': composer, 'generate': generate, 'melody':melody, 'velocity':melody}

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
def checkMidi(file: str, dataset, task, tag, file2='', padding=True):
    a = np.load(file)
    if padding:
        a = a.reshape(-1, 8)
    cnt = 0
    print(a.shape)
    a = list(a)
    f = 0
    index = random.randint(0, 50)
    for i in range(len(a)):
        if a[i][0] == 259:
            aa = a[f:i]
            if cnt == index:
                break
            cnt+=1
            f = i+1

    midi = encoding_to_MIDI(aa)
    # 保存 MIDI 字节到文件
    output_path = f'Data/output_test/{task}'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    midi.dump(output_path+f'/{dataset}_{tag}_{task}_sample.mid')
    
    if len(file2):
        b = np.load(file2)
        b = b.reshape(-1,8)
        print(b.shape)
        b = list(b)
        f = 0
        for i in range(len(b)):
            if b[i][0] == 259:
                bb = b[f:i]
                if cnt == index:
                    break
                cnt+=1
                f = i+1
        midib = encoding_to_MIDI(bb)
        midib.dump(output_path+f'/{dataset}_{tag}_{task}_sample_b.mid')

def checkPretrain():
    for d in default:
        for tag in ('train', 'test', 'valid'):
            for flag in ('', '_split'):
                print(d, tag, flag)
                file = f'Data\output_pretrain\{d}\{d}_{tag}{flag}.npy'
                a = np.load(file)
                print(a.shape)
                for num in range(8):                    
                    if flag == '_split':
                        m = a[:, :, num]
                        m = m.ravel()
                    else:
                        m = a[:, num]
                    print(f'{file}|{maps[num]}: {max(m)==mmaps[num]} and {mmaps[num]-1 in m}')
                    if num == 0:
                        print(m, np.count_nonzero(m == mmaps[0]))
            checkMidi(file, d, task='pretrain', tag=tag)
            
def checkFinetune(task):
    if task == 'generate':
        method = ['gen_method', 'pretrain_method'] 
    elif task == 'melody':
        method = ['default', 'simplified']
    else:
        method = ['',]
    for tag in ('train', 'test', 'valid'):
        for d in datasets[task]:
            for mthd in method:
                print(tag, d, mthd)
                file = f'Data\output_{task}\{d}\{mthd}\{d}_{tag}.npy'
                file2 = ''            
                a = np.load(file)
                print(a.shape)
                for num in range(8):
                    m = a[:, :, num]
                    m = m.ravel()
                    print(f'{file}|{maps[num]}: {max(m)==mmaps[num]} and {mmaps[num]-1 in m}')
                    if num == 0:
                        # 检查是否一个1024长度里只有一个eos
                        cnt = np.count_nonzero(m == mmaps[0])
                        print(m, cnt, cnt==a.shape[0])
                if mthd != 'pretrain_method':
                    ans = f'Data\output_{task}\{d}\{mthd}\{d}_{tag}_ans.npy'
                    b = np.load(ans)
                    print(b.shape, b.max(), b.min(), b[50:100])
                    if mthd == 'gen_method':
                        for num in range(8):
                            m = b[:, :, num]
                            m = m.ravel()
                            print(f'{file}|{maps[num]}: {max(m)==mmaps[num]} and {mmaps[num]-1 in m}')
                            if num == 0:
                                cnt = np.count_nonzero(m == mmaps[0])
                                print(m, cnt, cnt==b.shape[0])
                        file2 = ans
                if len(mthd):
                    task_m = task + f"_{mthd}"
                else:
                    task_m = task
                checkMidi(file, d, task_m, tag, file2)
                
# checkPretrain()
# checkFinetune('generate')
# checkFinetune('composer')
# checkFinetune('melody')
checkFinetune('velocity')


    
