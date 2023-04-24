import pickle
import math

bar_max = 256 # 歌曲最大的小节数

pos_resolution = 16  # 每个beat的分辨率（我们的程序将1个beat分成16份）
beat_note_factor = 4  # MIDI规定一个note由4拍（beat）组成
max_notes_per_bar = 2  # 一个小节（bar）最大的note数为2（4/4拍每小节为1 note，6/8拍每小节为0.75 note）

max_inst = 127 # 最大的乐器编号（0-127共128种乐器），编号128被设置为鼓

max_pitch = 127 # 最大的音高编号（0-127共128种音高）

duration_max = 8  # 最大持续时间：8个beat

velocity_quant = 4 #力度分辨率（每4个力度一组）
max_velocity = 127 #最大力度编号

max_ts_denominator = 6 #节奏类型n/m，规定：n<=2m && m<=2^6

#用于力度映射的一些数据
tempo_quant = 12  # 2 ** (1 / 12)
min_tempo = 16
max_tempo = 256


#构建数字与Midi事件间的对应关系（映射）
event2word = {'Bar': {}, 'Position': {}, 'Pitch': {}, 'Duration': {}, 'Velocity': {}, 'Instrument': {}, 'Tempo': {}, 'TimeSig':{} }
word2event = {'Bar': {}, 'Position': {}, 'Pitch': {}, 'Duration': {}, 'Velocity': {}, 'Instrument': {}, 'Tempo': {}, 'TimeSig':{} }

def special_tok(cnt, cls, f=None):

    event2word[cls][cls+' <PAD>'] = cnt
    print(cls+' <PAD>'+': ', cnt, file=f)
    word2event[cls][cnt] = cls+' <PAD>'
    cnt += 1

    event2word[cls][cls+' <MASK>'] = cnt
    print(cls + ' <MASK>' + ': ', cnt, file=f)
    word2event[cls][cnt] = cls+' <MASK>'
    cnt += 1

    event2word[cls][cls+' <SOS>'] = cnt
    print(cls + ' <SOS>' + ': ', cnt, file=f)
    word2event[cls][cnt] = cls+' <SOS>'
    cnt += 1

    event2word[cls][cls+' <EOS>'] = cnt
    print(cls + ' <EOS>' + ': ', cnt, file=f)
    word2event[cls][cnt] = cls+' <EOS>'
    cnt += 1

    event2word[cls][cls+' <CLS>'] = cnt
    print(cls + ' <CLS>' + ': ', cnt, file=f)
    word2event[cls][cnt] = cls+' <CLS>'
    cnt += 1

    event2word[cls][cls+' <SEP>'] = cnt
    print(cls + ' <SEP>' + ': ', cnt, file=f)
    word2event[cls][cnt] = cls+' <SEP>'
    cnt += 1

output_file="../dict.txt"
with open(output_file, 'w') as f:
    # Bar
    cnt, cls = 0, 'Bar'
    for i in range(bar_max):
        event2word[cls][f'Bar {i}'] = cnt
        print(f'Bar {i}: ', cnt, file=f)
        word2event[cls][cnt] = f'Bar {i}'
        cnt += 1
    special_tok(cnt, cls, f)

    # Position
    cnt, cls = 0, 'Position'
    for i in range(max_notes_per_bar * beat_note_factor *  pos_resolution): #每个bar内最大的可能位置数
        event2word[cls][f'Position {i}/{beat_note_factor *  pos_resolution}'] = cnt
        print(f'Position {i}/{beat_note_factor *  pos_resolution}: ', cnt, file=f)
        word2event[cls][cnt]= f'Position {i}/{beat_note_factor *  pos_resolution}'
        cnt += 1
    special_tok(cnt, cls, f)

    # Instrument
    cnt, cls = 0, 'Instrument'
    for i in range(max_inst+1):
        event2word[cls][f'Instrument {i}'] = cnt
        print(f'Instrument {i}: ', cnt, file=f)
        word2event[cls][cnt] = f'Instrument {i}'
        cnt += 1
    event2word[cls]['Instrument percussion'] = cnt # 将编号max_inst+1设为鼓
    print('Instrument percussion: ', cnt, file=f)
    word2event[cls][cnt] = 'Instrument percussion'
    cnt += 1
    special_tok(cnt, cls, f)

    # Pitch
    cnt, cls = 0, 'Pitch'
    for i in range(2 * max_pitch + 1 + 1): # 0~max_pitch是一般乐器，max_pitch+1~2*max_pitch+1是鼓的音高（平移后）
        if i<=max_pitch:
            event2word[cls][f'Pitch {i}'] = cnt
            print(f'Pitch {i}: ', cnt, file=f)
            word2event[cls][cnt] = f'Pitch {i}'
        else:
            event2word[cls][f'Pitch percussion {i-max_pitch-1}'] = cnt
            print(f'Pitch percussion {i-max_pitch-1}: ', cnt, file=f)
            word2event[cls][cnt] = f'Pitch percussion {i-max_pitch-1}'
        cnt += 1
    special_tok(cnt, cls, f)

    # Duration
    cnt, cls = 0, 'Duration'
    for i in range(duration_max * pos_resolution):
        event2word[cls][f'Duration {i}'] = cnt
        print(f'Duration {i}: ', cnt, file=f)
        word2event[cls][cnt] = f'Duration {i}'
        cnt += 1
    special_tok(cnt, cls, f)

    # Velocity
    def v2e(x):
        return x // velocity_quant
    def e2v(x):
        return (x * velocity_quant) + (velocity_quant // 2)
    cnt, cls = 0, 'Velocity'
    for i in range(v2e(max_velocity) + 1):
        event2word[cls][f'Velocity {e2v(i)}'] = cnt
        print(f'Velocity {e2v(i)}: ', cnt, file=f)
        word2event[cls][cnt] = f'Velocity {e2v(i)}'
        cnt += 1
    special_tok(cnt, cls, f)

    # TimeSig
    cnt, cls = 0, 'TimeSig'
    for i in range(0, max_ts_denominator + 1): # 节奏类型n/m，规定：n<=2m && m<=2^max_ts_denominator
        for j in range(1, ((2 ** i) * max_notes_per_bar) + 1):
            event2word[cls][f'TimeSig {j}/{2 ** i}'] = cnt
            print(f'TimeSig {j}/{2 ** i}: ', cnt, file=f)
            word2event[cls][cnt] = f'TimeSig {j}/{2 ** i}'
            cnt+=1
    special_tok(cnt, cls, f)

    # Tempo
    def b2e(x):
        x = max(x, min_tempo)
        x = min(x, max_tempo)
        x = x / min_tempo
        e = round(math.log2(x) * tempo_quant)
        return e
    def e2b(x):
        return 2 ** (x / tempo_quant) * min_tempo
    cnt, cls = 0, 'Tempo'
    for i in range(b2e(max_tempo) + 1):
        event2word[cls][f'Tempo {e2b(i)}'] = cnt
        print(f'Tempo {e2b(i)}: ', cnt, file=f)
        word2event[cls][cnt] = f'Tempo {e2b(i)}'
        cnt += 1
    special_tok(cnt, cls, f)

print(event2word)
print(word2event)
t = (event2word, word2event)

with open('../Octuple.pkl', 'wb') as f:
    pickle.dump(t, f)

