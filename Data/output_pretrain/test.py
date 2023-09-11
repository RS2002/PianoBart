import numpy as np
a=np.load("./POP909/POP909_train_split.npy")
for i in range(1024):
    print(a[110][i])
