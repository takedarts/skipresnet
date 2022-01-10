# What is SkipResNet ?

[<img alt="Architecture of DenseResNets" src="res/architecture.png" width="250px" align="right">](res/architecture.png)

SkipResNet is a Skip connected Residual convolutional neural Network for image recognition tasks.
Though an architecture of SkipResNets is a stack of Residual Blocks just like ResNets, each residual block has several inbounds from the previous blocks in same manner as DenseNets.
In order to improve the performance, a residual block in SkipResNets includes a Gate Module instead of element-wise additions in ResNets or concatenations in DenseNets.
A Gate Module contains an attention mechanism which selects useful features dynamically.
Experimental results indicate that an architecture of SkipResNets improves the performance in image classification tasks.

# What is DenseResNet ?

DenseResNet is a Densely connected Residual convolutional neural Network for image recognition tasks.
An architecture of DenseResNets is similar to SkipResNets, but the shortcut design is different.

<div class="clearfix"></div>

DenseResNets are published in a following paper:
1. Atsushi Takeda. "画像分類のためのDense Residual Networkの提案 (Dense Residual Networks for Image Classification)." The 23rd Meeting on Image Recognition and Understanding (MIRU2020), 2020 (in Japanese).

# How to use
## Dataset preparation
If you want to use the ImageNet dataset, you need to download the dataset archives and save them to `data/imagenet` (see [readme.txt](data/imagenet/readme.txt) for details). If you want to train a model with the CIFAR dataset, dataset preparation is not needed because the dataset will be downloaded automatically.

## Training
Run a training script which trains the model from scratch.
```
python src/train.py [config file] [output path]
```
For example, a following command trains a ResNet-110 model  using 2 GPUs with the CIFAR-100 dataset, and the results are saved in a output directory named `output_dir`.
```
python src/train.py \
    config/train/cifar100/ResNet-110.txt \
    output_directory \
    --gpus 0,1
```
This implementation supports training by using TPUs. A following command trains a ResNet-50 model using 8 TPUs with the ImageNet dataset loaded from Google Cloud Storage. In this case, you need to make shard files of the ImageNet dataset and stored them to Google Cloud Storage before starting the training.
```
PYTORCH_JIT=0 python src/train.py \
    configs/train/imagenet/ResNet-50.txt \
    output_directory \
    --tpus 8 \
    --data gs://<your backet>/data/imagenet
```

# Performances
The subscript of each model is the number of training runs, and the row indicates the median of the training results. For example, a row of "model<sub>(5)</sub>" shows the median performance of the 5 trained models.

## Performances of models trained on ImageNet-1k
Following models are trainied using the ImageNet-1k dataset from scratch. The image used for the training is cropped to a 224x224 size image, and no extra images are used.

### Models
|Model|# params|flops (224x224)|settings|
|:---|:---:|:---:|:---|
|ResNet-34<br>Skip-ResNet-34|21.80M<br>22.72M|3.681G<br>3.694G|[ResNet-34.txt](configs/train/imagenet/ResNet-34.txt)<br>[Skip-ResNet-34.txt](configs/train/imagenet/Skip-ResNet-34.txt)|
|ResNet-50<br>Skip-ResNet-50|25.56M<br>40.15M|4.138G<br>4.201G|[ResNet-50.txt](configs/train/imagenet/ResNet-50.txt)<br>[Skip-ResNet-50.txt](configs/train/imagenet/Skip-ResNet-50.txt)|
|ResNeXt-50-32x4d<br>Skip-ResNeXt-50-32x4d|25.03M<br>39.63M|4.292G<br>4.355G|[ResNeXt-50-32x4d.txt](configs/train/imagenet/ResNeXt-50-32x4d.txt)<br>[Skip-ResNeXt-50-32x4d.txt](configs/train/imagenet/Skip-ResNeXt-50-32x4d.txt)|

### Results on the ImageNet-1k dataset
|Model|top-1 acc.<br>(224x224)|top-1 acc.<br>(256x256)|top-1 acc.<br>(288x288)|top-1 acc.<br>(320x320)|top-1 acc.<br>(352x352)|top-1 acc.<br>(384x384)|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
|ResNet-34<sub>(5)</sub><br>Skip-ResNet-34<sub>(5)</sub>|0.7553<br>0.7675|0.7622<br>0.7759|0.7654<br>0.7778|0.7665<br>**0.7782**|0.7627<br>0.7751|0.7586<br>0.7704|
|ResNet-50<sub>(5)</sub><br>Skip-ResNet-50<sub>(5)</sub>|0.7901<br>0.8041|0.7953<br>0.8103|0.7964<br>**0.8120**|0.7954<br>0.8104|0.7926<br>0.8083|0.7885<br>0.8054|
|ResNeXt-50-32x4d<sub>(1)</sub><br>Skip-ResNeXt-50-32x4d<sub>(1)</sub>|0.7963<br>0.8067|0.8015<br>0.8125|0.8032<br>**0.8131**|0.8016<br>0.8110|<br>|<br>|

### Results on the ImageNet-A dataset
|Model|top-1 acc.<br>(224x224)|top-1 acc.<br>(256x256)|top-1 acc.<br>(288x288)|top-1 acc.<br>(320x320)|top-1 acc.<br>(352x352)|top-1 acc.<br>(384x384)|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
|ResNet-34<sub>(5)</sub><br>Skip-ResNet-34<sub>(5)</sub>|0.0143<br>0.0259|0.0204<br>0.0328|0.0297<br>0.0416|0.0321<br>0.0448|0.0364<br>**0.0453**|0.0408<br>0.0439|
|ResNet-50<sub>(5)</sub><br>Skip-ResNet-50<sub>(5)</sub>|0.0304<br>0.0695|0.0477<br>0.0889|0.0583<br>0.0987|0.0625<br>0.1015|0.0607<br>**0.1024**|0.0621<br>0.1015|

## Performances of models trained on CIFAR-10/100
Following models are trainied using the CIFAR-10/100 dataset from scratch. No extra images are used.

### Models
|Model|# params|flops (32x32)|settings|
|:---|:---:|:---:|:---|
|ResNet-110<br>Skip-ResNet-110|1.737M<br>2.189M|257.9M<br>265.4M|[ResNet-110.txt](configs/train/cifar100/ResNet-110.txt)<br>[Skip-ResNet-110.txt](configs/train/cifar100/Skip-ResNet-110.txt)|
|WideResNet-28-k10<br>Skip-WideResNet-28-k10|36.54M<br>38.18M|5.254G<br>5.266G|[WideResNet-28-k10.txt](configs/train/cifar100/WideResNet-28-k10.txt)<br>[Skip-WideResNet-28-k10.txt](configs/train/cifar100/Skip-WideResNet-28-k10.txt)|
|WideResNet-40-k10<br>Skip-WideResNet-40-k10|55.90M<br>58.64M|8.091G<br>8.111G|[WideResNet-40-k10.txt](configs/train/cifar100/WideResNet-40-k10.txt)<br>[Skip-WideResNet-40-k10.txt](configs/train/cifar100/Skip-WideResNet-40-k10.txt)|

### Results on the CIFAR-10/100 dataset
|Model|top-1 acc. (CIFAR-10)|top-1 acc (CIFAR-100)|
|:---|:---:|:---:|
|ResNet-110<sub>(5)</sub><br>Skip-ResNet-110<sub>(5)</sub>|0.9623<br>0.9660|0.7798<br>0.7988|
|WideResNet-28-k10<sub>(5)</sub><br>Skip-WideResNet-28-k10<sub>(5)</sub>|0.9787<br>0.9780|0.8425<br>0.8508|
|WideResNet-40-k10<sub>(5)</sub><br>Skip-WideResNet-40-k10<sub>(5)</sub>|0.9793<br>0.9792|0.8439<br>0.8498|

# Acknowledgement
This work is supported by JSPS KAKENHI Grant Number JP20K11871, and a part of this experiment is supported by the TPU Research Cloud program.
