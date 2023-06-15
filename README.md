# ML-2-HOTWORD-DETECTION

## Contents
1. [How to use](#how-to-use)
2. [Dataset](#dataset)
3. [Model training](#model-training)
4. [Sources](#sources)


## How to use
1. Install all requirements: `pip install -r requirements.txt`
2. Run application: `python -m src.cli <pipeline_name>`
    
Available pipeline names:
```
- prepare_dataset: Extracts one-second pieces of audio from given .wav-file and generate from them dataset. Put it into data/dataset
- train: Trains model for hot-word recognizing
- get_stream: Awaits while stream is not ready and then listen it and recognize hot-words in real-time
- static_audio_process: Recognizes hot-words from already saved audio file. 
```

## Dataset
Dataset have next structure:

~/data/dataset/wavs/ — folder with one-second pieces of audio

~/data/dataset/train.csv — train .csv file, contains paths to audio (relative of project root) and labels: have it hot-word or not

~/data/dataset/val.csv — as train.csv but for validation


## Model training
**Hyperparameters:**

| Parameter           |      Value       |
|:--------------------|:----------------:|
| Number of epochs    |        3         |
| Batch size          |        1         |
| Loss function       | CrossEntropyLoss |
| Optimizer           |      AdamW       |
| Scheduler           |  ExponentialLR   |
| Start learning rate |      1e-05       |
| Scheduler gamma     |       0.98       |

Train code was tested only on CPU, 
so it can have bugs on GPU.

## Sources
- [Dmitriy Menshikov](https://github.com/MenshikovDmitry) lecture
- [Pytorch Resnet18 source code](https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet18)