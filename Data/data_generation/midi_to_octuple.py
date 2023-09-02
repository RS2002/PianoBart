# Coding borrowed from 
# https://github.com/suryatmodulus/muzic/blob/22c838a4160e871c48c6575bbe082f0ab98094f0/musicbert/preprocess.py#L233

import os
import zipfile
import random
import miditoolkit
import math
import hashlib
import numpy as np
import re
dataset = input("Please input the dataset for generation: ")
data_path = f'Data/{dataset}'
task = input("Please input the task (pretrain/composer/generate) : ")
pad = True
if task == 'pretrain':
    pad = int(input("Padding (1/0):"))
    pad = True if pad == 1 else False
data_zip = zipfile.ZipFile(data_path+'.zip', 'r')
out_path = f'Data/output_{task}'
if not os.path.exists(out_path):
    os.mkdir(out_path)
out_path = os.path.join(out_path, dataset)
output_file = None
midi_dict = dict()


pos_resolution = 16  # per beat (quarter note)
max_bar = 255
velocity_quant = 4
tempo_quant = 12  # 2 ** (1 / 12)
min_tempo = 16
max_tempo = 256
duration_max = 8  # 2 ** 8 * beat
max_ts_denominator = 6  # x/1 x/2 x/4 ... x/64
max_notes_per_bar = 2  # 1/64 ... 128/64
beat_note_factor = 4  # In MIDI format a note is always 4 beats
deduplicate = True
filter_symbolic = False
filter_symbolic_ppl = 16
trunc_pos = 2 ** 16  # approx 30 minutes (1024 measures)
sample_len_max = 1000  # window length max
sample_overlap_rate = 4
max_window = 1024
ts_filter = False
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


# (0 Measure, 1 Pos, 2 Program, 3 Pitch, 4 Duration, 5 Velocity, 6 TimeSig, 7 Tempo)
# (Measure, TimeSig)
# (Pos, Tempo)
# Percussion: Program=128 Pitch=[128,255]


ts_dict = dict()
ts_list = list()
for i in range(0, max_ts_denominator + 1):  # 1 ~ 64
    for j in range(1, ((2 ** i) * max_notes_per_bar) + 1):
        ts_dict[(j, 2 ** i)] = len(ts_dict)
        ts_list.append((j, 2 ** i))
dur_enc = list()
dur_dec = list()
for i in range(duration_max):
    for j in range(pos_resolution):
        dur_dec.append(len(dur_enc))
        for k in range(2 ** i):
            dur_enc.append(len(dur_dec) - 1)
            
def t2e(x):
    assert x in ts_dict, 'unsupported time signature: ' + str(x)
    return ts_dict[x]


def e2t(x):
    return ts_list[x]


def d2e(x):
    return dur_enc[x] if x < len(dur_enc) else dur_enc[-1]


def e2d(x):
    return dur_dec[x] if x < len(dur_dec) else dur_dec[-1]


def v2e(x):
    return x // velocity_quant


def e2v(x):
    return (x * velocity_quant) + (velocity_quant // 2)


def b2e(x):
    x = max(x, min_tempo)
    x = min(x, max_tempo)
    x = x / min_tempo
    e = round(math.log2(x) * tempo_quant)
    return e


def e2b(x):
    return 2 ** (x / tempo_quant) * min_tempo

def get_hash(encoding):
    # add i[4] and i[5] for stricter match
    midi_tuple = tuple((i[2], i[3]) for i in encoding)
    midi_hash = hashlib.md5(str(midi_tuple).encode('ascii')).hexdigest()
    return midi_hash


def time_signature_reduce(numerator, denominator):
    # reduction (when denominator is too large)
    while denominator > 2 ** max_ts_denominator and denominator % 2 == 0 and numerator % 2 == 0:
        denominator //= 2
        numerator //= 2
    # decomposition (when length of a bar exceed max_notes_per_bar)
    while numerator > max_notes_per_bar * denominator:
        for i in range(2, numerator + 1):
            if numerator % i == 0:
                numerator //= i
                break
    return numerator, denominator

def writer(file_name, output_str_list):
    # note: parameter "file_name" is reserved for patching
    with open(output_file, 'a') as f:
        for output_str in output_str_list:
            f.write(output_str + '\n')

def MIDI_to_encoding(midi_obj):
    
    def time_to_pos(t):
        return round(t * pos_resolution / midi_obj.ticks_per_beat)
    
    notes_start_pos = [time_to_pos(j.start)
                        for i in midi_obj.instruments for j in i.notes]
    if len(notes_start_pos) == 0:
        return list()
    
    max_pos = min(max(notes_start_pos) + 1, trunc_pos) # 控制乐曲长度不超过trunc_pos
    
    
    pos_to_info = [[None for _ in range(4)] 
                    for _ in range(max_pos)]  # (Measure, TimeSig, Pos, Tempo)
    tsc = midi_obj.time_signature_changes
    tpc = midi_obj.tempo_changes
    for i in range(len(tsc)):
        for j in range(time_to_pos(tsc[i].time), time_to_pos(tsc[i + 1].time) if i < len(tsc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][1] = t2e(time_signature_reduce(
                    tsc[i].numerator, tsc[i].denominator))
    for i in range(len(tpc)):
        for j in range(time_to_pos(tpc[i].time), time_to_pos(tpc[i + 1].time) if i < len(tpc) - 1 else max_pos):
            if j < len(pos_to_info):
                pos_to_info[j][3] = b2e(tpc[i].tempo)
    for j in range(len(pos_to_info)):
        if pos_to_info[j][1] is None:
            # MIDI default time signature
            pos_to_info[j][1] = t2e(time_signature_reduce(4, 4))
        if pos_to_info[j][3] is None:
            pos_to_info[j][3] = b2e(120.0)  # MIDI default tempo (BPM)
    cnt = 0
    bar = 0
    measure_length = None
    for j in range(len(pos_to_info)):
        ts = e2t(pos_to_info[j][1])
        if cnt == 0:
            measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
        pos_to_info[j][0] = bar
        pos_to_info[j][2] = cnt
        cnt += 1
        if cnt >= measure_length:
            assert cnt == measure_length, 'invalid time signature change: pos = {}'.format(
                j)
            cnt -= measure_length
            bar += 1
    encoding = []
    start_distribution = [0] * pos_resolution
    for inst in midi_obj.instruments:
        for note in inst.notes:
            if time_to_pos(note.start) >= trunc_pos:
                continue
            start_distribution[time_to_pos(note.start) % pos_resolution] += 1
            info = pos_to_info[time_to_pos(note.start)]
            encoding.append((info[0], info[2], max_inst + 1 if inst.is_drum else inst.program, note.pitch + max_pitch +
                            1 if inst.is_drum else note.pitch, d2e(time_to_pos(note.end) - time_to_pos(note.start)), v2e(note.velocity), info[1], info[3]))
    if len(encoding) == 0:
        return list()
    tot = sum(start_distribution)
    start_ppl = 2 ** sum((0 if x == 0 else -(x / tot) *
                        math.log2((x / tot)) for x in start_distribution))
    # filter unaligned music
    if filter_symbolic:
        assert start_ppl <= filter_symbolic_ppl, 'filtered out by the symbolic filter: ppl = {:.2f}'.format(
            start_ppl)
    encoding.sort()
    return encoding



def encoding_to_MIDI(encoding):
    # TODO: filter out non-valid notes and error handling
    bar_to_timesig = [list()
                    for _ in range(max(map(lambda x: x[0], encoding)) + 1)]
    for i in encoding:
        bar_to_timesig[i[0]].append(i[6])
    bar_to_timesig = [max(set(i), key=i.count) if len(
        i) > 0 else None for i in bar_to_timesig]
    for i in range(len(bar_to_timesig)):
        if bar_to_timesig[i] is None:
            bar_to_timesig[i] = t2e(time_signature_reduce(
                4, 4)) if i == 0 else bar_to_timesig[i - 1]
    bar_to_pos = [None] * len(bar_to_timesig)
    cur_pos = 0
    for i in range(len(bar_to_pos)):
        bar_to_pos[i] = cur_pos
        try:
            ts = e2t(bar_to_timesig[i])
            measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
            cur_pos += measure_length
        except:
            continue
    pos_to_tempo = [list() for _ in range(
        cur_pos + max(map(lambda x: x[1], encoding)))]
    for i in encoding:
        try:
            pos_to_tempo[bar_to_pos[i[0]] + i[1]].append(i[7])
        except:
            continue
    pos_to_tempo = [round(sum(i) / len(i)) if len(i) >
                    0 else None for i in pos_to_tempo]
    for i in range(len(pos_to_tempo)):
        if pos_to_tempo[i] is None:
            pos_to_tempo[i] = b2e(120.0) if i == 0 else pos_to_tempo[i - 1]
    midi_obj = miditoolkit.midi.parser.MidiFile()

    def get_tick(bar, pos):
        return (bar_to_pos[bar] + pos) * midi_obj.ticks_per_beat // pos_resolution
    midi_obj.instruments = [miditoolkit.containers.Instrument(program=(
        0 if i == 128 else i), is_drum=(i == 128), name=str(i)) for i in range(128 + 1)]
    
    for i in encoding:
        start = get_tick(i[0], i[1])
        program = i[2]
        pitch = (i[3] - 128 if program == 128 else i[3])
        duration = get_tick(0, e2d(i[4]))
        if duration == 0:
            duration = 1
        end = start + duration
        velocity = e2v(i[5])
        try:
            midi_obj.instruments[program].notes.append(miditoolkit.containers.Note(
            start=start, end=end, pitch=pitch, velocity=velocity))
        except:
            continue
    midi_obj.instruments = [
        i for i in midi_obj.instruments if len(i.notes) > 0]
    cur_ts = None
    for i in range(len(bar_to_timesig)):
        new_ts = bar_to_timesig[i]
        if new_ts != cur_ts:
            try:
                numerator, denominator = e2t(new_ts)
            except:
                continue
            midi_obj.time_signature_changes.append(miditoolkit.containers.TimeSignature(
                numerator=numerator, denominator=denominator, time=get_tick(i, 0)))
            cur_ts = new_ts
    cur_tp = None
    for i in range(len(pos_to_tempo)):
        new_tp = pos_to_tempo[i]
        if new_tp != cur_tp:
            tempo = e2b(new_tp)
            midi_obj.tempo_changes.append(
                miditoolkit.containers.TempoChange(tempo=tempo, time=get_tick(0, i)))
            cur_tp = new_tp
    return midi_obj

def padding(file_name, e_segment, window=max_window):
    pad_num = window -  len(e_segment)
    if pad_num < 0:
        print('WARNING(LENGTH): ' + file_name + ' ' + 'The length of the music is longer than max window(1024).' + '\n', end='')
        e_segment = e_segment[:window-1]
        e_segment.append(tuple([i + 4 for i in token_boundary]))
        return e_segment
    for _ in range(pad_num):
        e_segment.append(tuple([i + 1 for i in token_boundary]))
    return e_segment

def F(file_name):
    # for test
    # midi_obj = miditoolkit.midi.parser.MidiFile(file_name)
    midi_obj = miditoolkit.midi.parser.MidiFile(f'{data_path}/{file_name}')
    midi_notes_count = sum(len(inst.notes) for inst in midi_obj.instruments)
    if midi_notes_count == 0:
        print('ERROR(BLANK): ' + file_name + '\n', end='')
        return None
    try:
        e = MIDI_to_encoding(midi_obj)
        if len(e) == 0:
            print('ERROR(BLANK): ' + file_name + '\n', end='')
            return None
        if ts_filter:
            allowed_ts = t2e(time_signature_reduce(4, 4))
            if not all(i[6] == allowed_ts for i in e):
                print('ERROR(TSFILT): ' + file_name + '\n', end='')
                return None
        if deduplicate:
            duplicated = False
            dup_file_name = ''
            midi_hash = '0' * 32
            try:
                midi_hash = get_hash(e)
            except BaseException as e:
                pass
            if midi_hash in midi_dict:
                dup_file_name = midi_dict[midi_hash]
                duplicated = True
            else:
                midi_dict[midi_hash] = file_name
            if duplicated:
                print('ERROR(DUPLICATED): ' + midi_hash + ' ' +
                        file_name + ' == ' + dup_file_name + '\n', end='')
                return None
        # sample_step = max(round(sample_len_max / sample_overlap_rate), 1)
        # e_segment = []
        # if task == 'pretrain':
            # e.insert(0, tuple([i + 3 for i in token_boundary]))
            # e_segment.append(tuple([i + 3 for i in token_boundary]))
        # for p in range(0 - random.randint(0, sample_len_max - 1), len(e), sample_step):
        #     L = max(p, 0)
        #     R = min(p + sample_len_max, len(e)) - 1
        #     bar_index_list = [e[i][0]
        #                     for i in range(L, R + 1) if e[i][0] is not None]
        #     bar_index_min = 0
        #     bar_index_max = 0
        #     if len(bar_index_list) > 0:
        #         bar_index_min = min(bar_index_list)
        #         bar_index_max = max(bar_index_list)
        #     offset_lower_bound = -bar_index_min
        #     offset_upper_bound = max_bar - 1 - bar_index_max
        #     # to make bar index distribute in [0, max_bar)
        #     bar_index_offset = random.randint(
        #         offset_lower_bound, offset_upper_bound) if offset_lower_bound <= offset_upper_bound else offset_lower_bound
        #     for i in e[L: R + 1]:
        #         if i[0] is None or i[0] + bar_index_offset < max_bar:
        #             t = list(i)
        #             t[0] += bar_index_offset
        #             e_segment.append(tuple(t))
        #         else:
        #             break
        # # no empty
        # if not all(len(i.split()) > tokens_per_note * 2 - 1 for i in output_str_list):
        #     print('ERROR(ENCODE): ' + file_name + ' ' + str(e) + '\n', end='')
        #     return False
        # e_segment.append(tuple([i + 4 for i in token_boundary]))
        # if task == 'generate':
        #     data_segment = e_segment[:len(e_segment) // 2]
        #     data_segment.append(tuple([i + 4 for i in token_boundary]))
        #     tag_segment = e_segment[len(e_segment) // 2:]
        #     data_segment = padding(file_name, data_segment)
        #     tag_segment = padding(file_name, tag_segment)
        #     print('SUCCESS: ' + file_name + '\n', end='')
        #     return data_segment, tag_segment
        # # if pad:
        # #     e_segment = padding(file_name, e_segment)
        # print('SUCCESS: ' + file_name + '\n', end='')
        # if task == 'composer':
        #     if dataset == 'asap':
        #         pattern = r"./(.*?)/."
        #         composer = re.search(pattern, file_name).group(1)
        #     elif dataset == 'Pianist8':
        #         composer = re.search(r"/([^/]+)/(.*?)/(.*?)_", file_name).group(2)
        #     return e_segment, composer
        # return e_segment
        # return True
        e_list = []
        # e.append(tuple([i + 4 for i in token_boundary]))
        flag = 1
        former = 0
        for i, ei in enumerate(e):
            if ei[0] > max_bar * flag:
                temp = list(e[former:i])
                if flag > 1:
                    for j, t in enumerate(temp):
                        tt = list(t)
                        tt[0] -= max_bar * (flag-1) +1
                        temp[j] = tuple(tt)
                temp.append(tuple([i + 4 for i in token_boundary]))
                e_list.append(temp)
                former = i
                flag += 1
                # e = e[:i]
        temp = list(e[former:])
        if flag > 1:
            for j, t in enumerate(temp):
                tt = list(t)
                tt[0] -= max_bar * (flag-1) +1
                temp[j] = tuple(tt)
        temp.append(tuple([i + 4 for i in token_boundary]))
        e_list.append(temp)
        print(e[0], e[-2], e[-1], len(e), len(e_list))
        output_list = []
        for ei in e_list:
            print(ei[0], ei[-2], ei[-1], len(ei))
            if task == 'generate':
                half = max_window-1 if len(ei) >= 2 * max_window else len(ei) // 2 - 1
                data_segment = ei[:half]
                for i, ds in enumerate(data_segment):
                    if ds[0] >= data_segment[-1][0]:
                        break
                data_segment = ei[:i]
                last = data_segment[-1][0]
                data_segment.append(tuple([i + 4 for i in token_boundary]))
                data_segment = padding(file_name, data_segment)
                tag_segment = ei[i:]
                print(len(ei), len(data_segment), len(tag_segment))
                tag_segment = padding(file_name, tag_segment)
                print(data_segment[0], last, data_segment[-2], data_segment[-1], len(data_segment))
                print(tag_segment[0], tag_segment[-2], tag_segment[-1], len(tag_segment))
                print('SUCCESS: ' + file_name + '\n', end='')
                output_list.append((data_segment, tag_segment))
                # return data_segment, tag_segment
            if task == 'pretrain':
                if pad:
                    ei = padding(file_name, ei)
                    print(ei[0], ei[-2], ei[-1], len(ei))
                output_list.append(ei)
            print('SUCCESS: ' + file_name + '\n', end='')
            if task == 'composer':
                ei = padding(file_name, ei)
                print(ei[0], ei[-2], ei[-1], len(ei))
                if dataset == 'asap':
                    pattern = r"./(.*?)/."
                    composer = re.search(pattern, file_name).group(1)
                elif dataset == 'Pianist8':
                    composer = re.search(r"/([^/]+)/(.*?)/(.*?)_", file_name).group(2)
                print(composer)
                output_list.append((ei,composer))
                # return e, composer
        return output_list
        return True

    except BaseException as ex:
        print('ERROR(PROCESS): ' + file_name + ' ' + str(ex) + '\n', end='')
        return False
    print('ERROR(GENERAL): ' + file_name + '\n', end='')
    return False

def G_composer(file_name, output: list, ans: list):
    try:
        ret = F(file_name)
        
        # ret, comp = F(file_name)
        if ret:
            for seq, comp in ret:
                output.append(seq)
                ans.append(comp)            
            return True
    except BaseException as e:
        print('ERROR(UNCAUGHT): ' + file_name + '\n', end='')
        return False


def G_generate(file_name, output: list, ans: list):
    try:
        ret = F(file_name)
        # ret, tag = F(file_name)
        if ret:
            for seq, tag in ret:
                output.append(seq)
                ans.append(tag)            
            return True
    except BaseException as e:
        print('ERROR(UNCAUGHT): ' + file_name + '\n', end='')
        return False

def G(file_name, output: list):
    try:
        ret = F(file_name)
        if ret:
            for seq in ret:
                if pad:
                    output.append(seq)
                else:
                    output += seq
            return True
    except BaseException as e:
        print('ERROR(UNCAUGHT): ' + file_name + '\n', end='')
        return False

def data_split(data: np.array):
    m = data.shape[0] // max_window + 1
    pad_num = m * max_window - data.shape[0]
    # padding
    padded = np.append(data, [[i + 1 for i in token_boundary]]*pad_num, axis=0)
    return padded.reshape(m, max_window, tokens_per_note)


    
if __name__ == '__main__':
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if dataset == 'asap':
        comp_path = os.path.join(data_path, 'asap')
    elif dataset == 'Pianist8':
        comp_path = os.path.join(data_path, 'Pianist8', 'midi')
    file_list = [n for n in data_zip.namelist() if n[-4:].lower()
                    == '.mid' or n[-5:].lower() == '.midi']
    random.shuffle(file_list)
    if task == 'composer':
        composers = list(set([f.name for f in os.scandir(comp_path) if f.is_dir() and f.name != 'util']))
        encoding_map = {string: index for index, string in enumerate(composers)}
        json_file_path = f"{out_path}/{dataset}_{task}.json"
        with open(json_file_path, "w") as json_file:
            import json
            json.dump(encoding_map, json_file, indent=4)
            
    ok_cnt = 0
    all_cnt = 0
    for sp in ['test','train', 'valid']:
        print(sp)
        total_file_cnt = len(file_list)
        file_list_split = []
        if sp == 'train':  # 80%
            file_list_split = file_list[: 80 * total_file_cnt // 100]
        if sp == 'valid':  # 10%
            file_list_split = file_list[80 * total_file_cnt //
                                        100: 90 * total_file_cnt // 100]
        if sp == 'test':  # 10%
            file_list_split = file_list[90 * total_file_cnt // 100:]
        output_file = '{}/{}_{}.npy'.format(out_path, dataset, sp)
        split_file = '{}/{}_{}_split.npy'.format(out_path, dataset, sp)
        res = []
        output = []
        ans = []
        for mid in file_list_split:
            if task == 'composer':
                res.append(G_composer(mid, output, ans))
            elif task == 'pretrain':
                res.append(G(mid, output))
            elif task == 'generate':
                res.append(G_generate(mid, output, ans))
        all_cnt += sum((1 if i is not None else 0 for i in res))
        ok_cnt += sum((1 if i is True else 0 for i in res))
        output = np.array(output)
        np.save(output_file, output)
        if task == 'pretrain':
            if not pad:
                output_split = data_split(output)
                np.save(split_file, output_split)
        elif task == 'composer':
            for i, comp in enumerate(ans):
                ans[i] = encoding_map[comp]
            ans = np.array(ans)
            ans_file = f'{out_path}/{dataset}_{sp}_{task[:3]}ans.npy'
            np.save(ans_file, ans)
        elif task == 'generate':
            ans = np.array(ans)
            ans_file = f'{out_path}/{dataset}_{sp}_{task[:3]}ans.npy'
            np.save(ans_file, ans)
        # output_file = None
    print('{}/{} ({:.2f}%) MIDI files successfully processed'.format(ok_cnt,
                                                                     all_cnt, ok_cnt / all_cnt * 100))