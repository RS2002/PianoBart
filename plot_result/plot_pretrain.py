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

def plot_result(PianoBART,sample,MusicBERT,MidiBERT,task="",title=""):
    plt.plot(PianoBART[task], 'r',label="PianoBART")
    # plt.plot(sample[task], 'g--',label="PianoBART-sample")
    plt.plot(MusicBERT[task], 'b',label="MusicBERT")
    plt.plot(MidiBERT[task], 'y',label="MidiBERT")
    plt.title(title)
    plt.legend()
    plt.show()



PianoBART=read_from_pretrain("./pretrain_result/PianoBART")
sample=read_from_pretrain("./pretrain_result/sample")
MusicBERT=read_from_pretrain("./pretrain_result/MusicBERT")
MidiBERT=read_from_pretrain("./pretrain_result/MidiBERT")

plot_result(PianoBART,sample,MusicBERT,MidiBERT,task="loss_train",title="Loss of Train")
plot_result(PianoBART,sample,MusicBERT,MidiBERT,task="loss_test",title="Loss of Test")
plot_result(PianoBART,sample,MusicBERT,MidiBERT,task="acc_train",title="Accuracy of Train")
plot_result(PianoBART,sample,MusicBERT,MidiBERT,task="acc_test",title="Accuracy of Test")