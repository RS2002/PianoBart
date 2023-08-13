import numpy as np

def split(data_list,label_list,prob_list):
    data=None
    label=None
    for i in range(len(data_list)):
        x=np.load(data_list[i])
        y=np.load(label_list[i])
        #print(x.shape)
        if data is None:
            data=x
            label=y
        else:
            data=np.concatenate([data,x])
            label=np.concatenate([label,y])
    #return
    n=len(data)
    previous=0
    for i in range(len(data_list)):
        temp=previous+int(n*prob_list[i])
        if i==len(data_list)-1:
            temp=n
        np.save(data_list[i], data[previous:temp])
        np.save(label_list[i], label[previous:temp])
        previous=temp


# split(
#     ["./asap/asap_train.npy","./asap/asap_valid.npy","./asap/asap_test.npy"],
#     ["./asap/asap_train_comans.npy","./asap/asap_valid_comans.npy","./asap/asap_test_comans.npy"],
#     [0.8,0.1,0.1]
# )
split(
    ["./Pianist8/Pianist8_train.npy","./Pianist8/Pianist8_valid.npy","./Pianist8/Pianist8_test.npy"],
    ["./Pianist8/Pianist8_train_comans.npy","./Pianist8/Pianist8_valid_comans.npy","./Pianist8/Pianist8_test_comans.npy"],
    [0.8,0.1,0.1]
)