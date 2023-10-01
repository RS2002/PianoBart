from matplotlib import pyplot as plt
import re
import numpy as np

pattern = '\d+\.\d*'


def read_from_pretrain(path):
    loss_train=[]
    loss_test=[]
    acc_train=[]
    acc_test=[]
    with open(path) as f:
        for line in f:
            match_obj = re.findall(pattern, line)
            for i in range(len(match_obj)):
                match_obj[i] = eval(match_obj[i])
            if len(match_obj)==18:
                loss_train.append(match_obj[0])
                loss_test.append(match_obj[9])
                acc_train.append(np.average(match_obj[1:9]))
                acc_test.append(np.average(match_obj[10:]))
            elif len(match_obj)==10:
                loss_train.append(match_obj[0])
                loss_test.append(match_obj[5])
                acc_train.append(np.average(match_obj[1:5]))
                acc_test.append(np.average(match_obj[6:]))
    result={
        "loss_train":loss_train,
        "loss_test":loss_test,
        "acc_train":acc_train,
        "acc_test":acc_test
    }
    return result

def plot_result(PianoBART,MusicBERT,MidiBERT,PianoBART_ablation,MusicBERT_ablation,MidiBERT_ablation,task="",title=""):
    linewidth=2.5
    plt.plot(PianoBART[task], 'r',label="PianoBART", linewidth=linewidth)
    plt.plot(MusicBERT[task], 'purple',label="MusicBERT", linewidth=linewidth)
    plt.plot(MidiBERT[task], 'g',label="MidiBERT", linewidth=linewidth)
    plt.plot(PianoBART_ablation[task], 'b',label="PianoBART Ablation", linewidth=linewidth)
    plt.plot(MusicBERT_ablation[task], 'y',label="MusicBERT Ablation", linewidth=linewidth)
    plt.plot(MidiBERT_ablation[task], 'orange',label="MidiBERT Ablation", linewidth=linewidth)
    plt.xlabel("Epoch", fontsize=16)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.title(title, fontsize=18)
    plt.legend(prop={'size': 18}, framealpha=0.3)
    plt.show()



PianoBART=read_from_pretrain("./pretrain_result/PianoBART")
MusicBERT=read_from_pretrain("./pretrain_result/MusicBERT")
MidiBERT=read_from_pretrain("./pretrain_result/MidiBERT")
PianoBART_ablation=read_from_pretrain("./pretrain_result0/PianoBART")
MusicBERT_ablation=read_from_pretrain("./pretrain_result0/MusicBERT")
MidiBERT_ablation=read_from_pretrain("./pretrain_result0/MidiBERT")


plot_result(PianoBART,MusicBERT,MidiBERT,PianoBART_ablation,MusicBERT_ablation,MidiBERT_ablation,task="loss_train",title="Loss of Train")
plot_result(PianoBART,MusicBERT,MidiBERT,PianoBART_ablation,MusicBERT_ablation,MidiBERT_ablation,task="loss_test",title="Loss of Test")
plot_result(PianoBART,MusicBERT,MidiBERT,PianoBART_ablation,MusicBERT_ablation,MidiBERT_ablation,task="acc_train",title="Accuracy of Train")
plot_result(PianoBART,MusicBERT,MidiBERT,PianoBART_ablation,MusicBERT_ablation,MidiBERT_ablation,task="acc_test",title="Accuracy of Test")