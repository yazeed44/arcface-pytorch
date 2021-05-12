# Arcface-pytorch 
This is my own implementation for Arcface to be used for deep face recognition, as listed in this paper.\
[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698v3.pdf)

## Setting up environment
```shell
conda create --name arcface_pytorch --file requirements.txt
```

### Setting up training dataset
Download [CASIA-WebFace](https://github.com/happynear/AMSoftmax/issues/18) and [CACD2000](https://bcsiriuschen.github.io/CARC/), and extract them inside raw_datasets.
The directory should look like this
```
raw_datasets
│   casia_webface_labels.txt
└───CACD
│   │   14_Aaron_Johnson_0001.jpg
│   │   14_Aaron_Johnson_0002.jpg
│   │   .....
└───CASIA-WebFace
│   └───0000045
│       │   001.jpg
│       │   002.jpg
│       │   ...
│   └───0000099
│       │   001.jpg
│       │   002.jpg
│       │   ...  
│   └───.......
```
- Make a directory in the root directory called `datasets`
- Run `python pre_processing.py`
- Wait until it is done. The result directory `dataset` should look like this:
```
datasets
│   └───50 Cent
│       │   001_normal.jpg
│       │   002_horizontal.jpg
│       │   ...
│   └───...
```
## Training
- Create `models` directory in project root
- You can change training settings from `settings.py`
- To start training, run `python train.py`
- Once training is done, you will see the model in `models` directory

## References
[ArcFace: Additive Angular Margin Loss for Deep Face Recognition ](https://arxiv.org/pdf/1801.07698v3.pdf)
[A Discriminative Feature Learning Approachfor Deep Face Recognition](https://ydwen.github.io/papers/WenECCV16.pdf)  
[Ring loss: Convex Feature Normalization for Face Recognition](https://arxiv.org/pdf/1803.00130.pdf)  
[InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)  
[focal-loss.pytorch](https://github.com/mbsariyildiz/focal-loss.pytorch)  
[insightface](https://github.com/deepinsight/insightface)