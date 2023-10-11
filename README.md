# PianoBart

**Authors:** Zijian Zhao, Weichao Zeng, Fupeng He, Yutong He, Yiyi Wang

**Supervisors:** Chengying Gao, Xiao Liang

**Article:** Zijian Zhao, Weichao Zeng, Fupeng He, Yutong He, Yiyi Wang, Xiao Liang, Chengying Gao*: PianoBART: Symbolic Piano Music Understanding and Generating with Large-Scale Pre-Training (in progress)

Some parts of our code borrows from [muzic/musicbert at main · microsoft/muzic (github.com)](https://github.com/microsoft/muzic/tree/main/musicbert) and [wazenmai/MIDI-BERT: This is the official repository for the paper, MidiBERT-Piano: Large-scale Pre-training for Symbolic Music Understanding. (github.com)](https://github.com/wazenmai/MIDI-BERT).



## 1. Dataset

**Pretrain:** POP1K7, ASAP, POP909, Pianist8, EMOPIA

**Generation:** Maestro, GiantMidi

**Composer Classification:** ASAP, Pianist8

**Emotion Classification:** EMOPIA

**Velocity Prediction:** GiantMidi

**Melody Prediction:** POP909



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
    #abalation()
    #ablation_eval
```

You can uncomment the corresponding function to perform the desired task.



### 2.2 Pretrain

Uncomment the “pretrain()” in main.py and run it.

```shell
python main.py
```



### 2.3 Finetune


#### 2.3.1 Generation

> Note: Before run the code, please do the following steps to patch the code.

1. Locate the file of `shapesimilarity.py`, which probably is in the path of `your_env/lib/python{version}/site-packages/shapesimilarity/shapesimilarity.py`.

2. Use the patch we provide, simply just run the following command in the terminal.

```shell
patch {path of shapesimilarity.py} < patches/shapesimilarity.patch
```



Uncomment the “finetune_generation()” in main.py and run it.

```shell
python main.py
```



#### 2.3.2 Composer Classification

Uncomment the “finetune()” in main.py and run it.

```

```



#### 2.3.3 Emotion Classification

Uncomment the “finetune()” in main.py and run it.

```

```



#### 2.3.4 Velocity Prediction

Uncomment the “finetune()” in main.py and run it.

```

```



#### 2.3.5 Melody Prediction

Uncomment the “finetune()” in main.py and run it.

```

```



## 3. Demo

In this section, you can input an intro (MIDI file) to PianoBart, and it will generate a new MIDI file inspired by the input. Simply provide the intro as input, and PianoBart will use its trained models to generate a new MIDI file with a similar style and tone.

```shell
python --ckpt <model path> --input <input path> --output <output path> demo.py
```

