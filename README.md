# PianoBart

**Authors:** Zijian Zhao, Weichao Zeng, Fupeng He, Yutong He, Yiyi Wang

**Supervisors:** Chengying Gao, Xiao Liang

**Article:**  under way

Some parts of our code borrows from [muzic/musicbert at main · microsoft/muzic (github.com)](https://github.com/microsoft/muzic/tree/main/musicbert) and [wazenmai/MIDI-BERT: This is the official repository for the paper, MidiBERT-Piano: Large-scale Pre-training for Symbolic Music Understanding. (github.com)](https://github.com/wazenmai/MIDI-BERT).



## 1. Dataset

### 1.1 Pretrain

Our pretraining dataset includes POP1K7, ASAP, POP909, Pianist8, and EMOPIA.



### 1.2 Others

We have not yet collected enough data for downstream tasks.



## 2. How to run the model

### 2.1 About the environment

We provide a conda-based environment. To use this environment, please install it using the following command:

```shell
conda env create -f environment.yml
```

This environment has been tested and is working properly.

To run the model, please refer to the code at the bottom of "main.py", which is shown as follows.

```python
if __name__ == '__main__':
    pretrain()
    #finetune()
    #eval()
    #finetune_generation()
    #finetune_eval()
```

You can uncomment the corresponding function to perform the desired task.



### 2.2 Pretrain



### 2.3 Finetune





## 3. Demo

A demo is currently under development.

In this section, you can input an intro (MIDI file) to PianoBart, and it will generate a new MIDI file inspired by the input. Simply provide the intro as input, and PianoBart will use its trained models to generate a new MIDI file with a similar style and tone.

