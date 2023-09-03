import numpy as np
a=np.load("midi_test_split.npy")
for i in range(1024):
    print(a[4,i,0])