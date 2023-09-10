import numpy as np
a=np.load("./maestro_train.npy")
b=np.load("./maestro_train_genans.npy")
for i in range(1024):
    print(a[11][i])
