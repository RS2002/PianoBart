import numpy as np
a=np.load("./maestro_train.npy")
b=np.load("./maestro_train_ans.npy")
for i in range(1024):
    print(b[0][i])