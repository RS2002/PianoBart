import numpy as np
a=np.load("./asap/asap_test.npy")
for i in range(1024):
    print(a[2,i,0])