import numpy as np
import pickle




#TODO:通过PianoBart输入Midi intro，输出续写的Midi文件

def Octuple2MIDI(Octuple_list, w2e, is_CP=False):
    #TODO
    pass




#测试
def test():
    with open("./Data/Octuple.pkl", 'rb') as f:
        e2w, w2e = pickle.load(f)
    test_Octuple = np.load("./Data/output/midi_split.npy")[0]
    result=Octuple2MIDI(test_Octuple,w2e)



if __name__ == '__main__':
    test()