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
python src/train.py \
    configs/train/imagenet/ResNet-50.txt \
    output_directory \
    --tpus 8 \
    --data gs://<your backet>/data/imagenet
```

# Performances
The subscript of each model is the number of training runs, and the row indicates the median of the training results. For example, a row of "model<sub>(5)</sub>" shows the median performances of the 5 trained models.
## Performances of models trained on ImageNet-1k
### Models
|Model|# params|flops (224x224)|settings|
|:---|:---:|:---:|:---|
|ResNet-34<br>Skip-ResNet-34|21.8M<br>22.7M|3.68G<br>3.69G|[ResNet-34.txt](configs/train/imagenet/ResNet-34.txt)<br>[Skip-ResNet-34.txt](configs/train/imagenet/Skip-ResNet-34.txt)|
|ResNet-50<br>Skip-ResNet-50|25.6M<br>40.2M|4.14G<br>4.20G|[ResNet-50.txt](configs/train/imagenet/ResNet-50.txt)<br>[Skip-ResNet-50.txt](configs/train/imagenet/Skip-ResNet-50.txt)|
### Results on the ImageNet-1k dataset
|<br>Model|top-1 acc.<br>(224x224)|top-1 acc.<br>(256x256)|top-1 acc.<br>(288x288)|
|:---|:---:|:---:|:---:|
|ResNet-34<sub>(1)</sub><br>Skip-ResNet-34<sub>(1)</sub>|0.7539<br>0.7675|0.7612<br>0.7759|0.7636<br>0.7614|
|ResNet-50<sub>(1)</sub><br>Skip-ResNet-50<sub>(1)</sub>|0.7887<br>0.8029|0.7948<br>0.8103|0.7954<br>0.8120|

## Performances of models trained on CIFAR-100
### Models
|Model|# params|flops (224x224)|settings|
|:---|:---:|:---:|:---|
|ResNet-110<br>Skip-ResNet-34|1.74M<br>2.19M|258M<br>265M|[ResNet-110.txt](configs/train/cifar100/ResNet-110.txt)<br>[Skip-ResNet-110.txt](configs/train/cifar100/Skip-ResNet-110.txt)|
|WideResNet-28-k10<br>Skip-WideResNet-28-k10|36.5M<br>38.2M|5.25G<br>5.27G|[WideResNet-28-k10.txt](configs/train/cifar100/WideResNet-28-k10.txt)<br>[Skip-WideResNet-28-k10.txt](configs/train/cifar100/Skip-WideResNet-28-k10.txt)|
|WideResNet-40-k10<br>Skip-WideResNet-40-k10|55.9M<br>58.6M|8.09G<br>8.11G|[WideResNet-40-k10.txt](configs/train/cifar100/WideResNet-40-k10.txt)<br>[Skip-WideResNet-40-k10.txt](configs/train/cifar100/Skip-WideResNet-40-k10.txt)|
<!--
### CIFAR-100
|Model|# params|flops|top-1 acc.|settings|
|---:|:---:|:---:|:---:|:---|
|ResNet-110<br>Dense-ResNet-110|1.74M<br>2.23M|258M<br>264M|79.03%<br>80.34%|[resnet-110.txt](config/cifar/resnet-110.txt)|

### CIFAR-10
|Model|# params|flops|top-1 acc.|settings|
|---:|:---:|:---:|:---:|:---|
|ResNet-110<br>Dense-ResNet-110|1.74M<br>2.23M|258M<br>264M|96.40%<br>96.59%|[resnet-110.txt](config/cifar/resnet-110.txt)|
-->
