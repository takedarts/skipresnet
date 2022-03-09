# What is SkipResNet ?

[<img alt="Architecture of DenseResNets" src="res/architecture.png" width="250px" align="right">](res/architecture.png)

SkipResNet is a Skip connected Residual convolutional neural Network for image recognition tasks.
Though an architecture of SkipResNets is a stack of Residual Blocks just like ResNets, each residual block has several inbounds from the previous blocks in the same manner as DenseNets.
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
|Model|# of params|flops (224x224)|settings|
|:---|:---:|:---:|:---|
|ResNet-34<br>Skip-ResNet-34|21.80M<br>22.72M|3.681G<br>3.694G|[ResNet-34.txt](configs/train/imagenet/ResNet-34.txt)<br>[Skip-ResNet-34.txt](configs/train/imagenet/Skip-ResNet-34.txt)|
|ResNet-50<br>Skip-ResNet-50|25.56M<br>40.15M|4.138G<br>4.201G|[ResNet-50.txt](configs/train/imagenet/ResNet-50.txt)<br>[Skip-ResNet-50.txt](configs/train/imagenet/Skip-ResNet-50.txt)|
|ResNet-101<br>Skip-ResNet-101|44.55M<br>83.36M|7.874G<br>8.017G|[ResNet-101.txt](configs/train/imagenet/ResNet-101.txt)<br>[Skip-ResNet-101.txt](configs/train/imagenet/Skip-ResNet-101.txt)|
|ResNeXt-50-32x4d<br>Skip-ResNeXt-50-32x4d|25.03M<br>39.63M|4.292G<br>4.355G|[ResNeXt-50-32x4d.txt](configs/train/imagenet/ResNeXt-50-32x4d.txt)<br>[Skip-ResNeXt-50-32x4d.txt](configs/train/imagenet/Skip-ResNeXt-50-32x4d.txt)|
|ResNeXt-101-32x4d<br>Skip-ResNeXt-101-32x4d|44.18M<br>82.99M|8.063G<br>8.205G|[ResNeXt-101-32x4d.txt](configs/train/imagenet/ResNeXt-101-32x4d.txt)<br>[Skip-ResNeXt-101-32x4d.txt](configs/train/imagenet/Skip-ResNeXt-101-32x4d.txt)|
|RegNetY-1.6<br>Skip-RegNetY-1.6|11.20M<br>14.76M|1.650G<br>1.677G|[RegNetY-1.6.txt](configs/train/imagenet/RegNetY-1.6.txt)<br>[Skip-RegNetY-1.6.txt](configs/train/imagenet/Skip-RegNetY-1.6.txt)|
|RegNetY-3.2<br>Skip-RegNetY-3.2|19.44M<br>25.35M|3.229G<br>3.265G|[RegNetY-3.2.txt](configs/train/imagenet/RegNetY-3.2.txt)<br>[Skip-RegNetY-3.2.txt](configs/train/imagenet/Skip-RegNetY-3.2.txt)|
|ConvNeXt-T<br>Skip-ConvNeXt-T|28.59M<br>31.14M|4.569G<br>4.591G|[ConvNeXt-T.txt](configs/train/imagenet/ConvNeXt-T.txt)<br>[Skip-ConvNeXt-T.txt](configs/train/imagenet/Skip-ConvNeXt-T.txt)|
|ConvNeXt-S<br>Skip-ConvNeXt-S|50.22M<br>56.58M|8.863G<br>8.912G|[ConvNeXt-S.txt](configs/train/imagenet/ConvNeXt-S.txt)<br>[Skip-ConvNeXt-S.txt](configs/train/imagenet/Skip-ConvNeXt-S.txt)|


### Results on the ImageNet-1k dataset
|Model|top-1 acc.<br>(224x224)|top-1 acc.<br>(256x256)|top-1 acc.<br>(288x288)|top-1 acc.<br>(320x320)|
|:---|:---:|:---:|:---:|:---:|
|ResNet-34<sub>(3)</sub><br>Skip-ResNet-34<sub>(3)</sub>|0.7553<br>0.7675|0.7622<br>0.7759|0.7654<br>0.7778|0.7665<br>**0.7782**|
|ResNet-50<sub>(5)</sub><br>Skip-ResNet-50<sub>(5)</sub>|0.7901<br>0.8041|0.7953<br>0.8103|0.7964<br>**0.8120**|0.7954<br>0.8104|
|ResNet-101<sub>(3)</sub><br>Skip-ResNet-101<sub>(3)</sub>|0.8036<br>0.8139|0.8100<br>0.8217|0.8095<br>**0.8234**|0.8086<br>0.8208|
|ResNeXt-50-32x4d<sub>(3)</sub><br>Skip-ResNeXt-50-32x4d<sub>(3)</sub>|0.7973<br>0.8067|0.8015<br>0.8125|0.8030<br>**0.8131**|0.8011<br>0.8126|
|ResNeXt-101-32x4d<sub>(3)</sub><br>Skip-ResNeXt-101-32x4d<sub>(3)</sub>|0.8066<br>0.8139|0.8102<br>0.8203|0.8112<br>**0.8216**|0.8101<br>0.8210|
|RegNetY-1.6<sub>(3)</sub><br>Skip-RegNetY-1.6<sub>(3)</sub>|0.7736<br>0.7794|0.7841<br>0.7887|0.7879<br>0.7936|0.7904<br>**0.7946**|
|RegNetY-3.2<sub>(3)</sub><br>Skip-RegNetY-3.2<sub>(3)</sub>|0.7849<br>0.7884|0.7933<br>0.7960|0.7974<br>0.7997|0.7981<br>**0.8000**|
|ConvNeXt-T<sub>(3)</sub><br>Skip-ConvNeXt-T<sub>(3)</sub>|0.8157<br>0.8158|0.8171<br>0.8205|0.8157<br>**0.8224**|0.8094<br>**0.8224**|
|ConvNeXt-S<sub>(3)</sub><br>Skip-ConvNeXt-S<sub>(3)</sub>|0.8314<br>0.8333|0.8344<br>0.8367|0.8341<br>0.8374|0.8307<br>**0.8375**|

### Results on the ImageNet-A dataset
|Model|top-1 acc.<br>(224x224)|top-1 acc.<br>(256x256)|top-1 acc.<br>(288x288)|top-1 acc.<br>(320x320)|
|:---|:---:|:---:|:---:|:---:|
|ResNet-34<sub>(3)</sub><br>Skip-ResNet-34<sub>(3)</sub>|0.0143<br>0.0259|0.0204<br>0.0328|0.0297<br>0.0416|0.0321<br>**0.0448**|
|ResNet-50<sub>(5)</sub><br>Skip-ResNet-50<sub>(5)</sub>|0.0304<br>0.0695|0.0477<br>0.0889|0.0583<br>0.0987|0.0625<br>**0.1015**|
|ResNet-101<sub>(3)</sub><br>Skip-ResNet-101<sub>(3)</sub>|0.0635<br>0.1157|0.0869<br>0.1324|0.1015<br>**0.1481**|0.1023<br>0.1455|
|ResNeXt-50-32x4d<sub>(3)</sub><br>Skip-ResNeXt-50-32x4d<sub>(3)</sub>|0.0537<br>0.0916|0.0743<br>0.1072|0.0844<br>**0.1179**|0.0843<br>0.1179|
|ResNeXt-101-32x4d<sub>(3)</sub><br>Skip-ResNeXt-101-32x4d<sub>(3)</sub>|0.0889<br>0.1319|0.1155<br>0.1528|0.1261<br>**0.1628**|0.1288<br>0.1557|
|RegNetY-1.6<sub>(3)</sub><br>Skip-RegNetY-1.6<sub>(3)</sub>|0.0376<br>0.0520|0.0493<br>0.0649|0.0580<br>0.0748|0.0621<br>**0.0779**|
|RegNetY-3.2<sub>(3)</sub><br>Skip-RegNetY-3.2<sub>(3)</sub>|0.0504<br>0.0599|0.0616<br>0.0720|0.0709<br>0.0775|0.0752<br>**0.0821**|
|ConvNeXt-T<sub>(3)</sub><br>Skip-ConvNeXt-T<sub>(3)</sub>|0.1032<br>0.1231|0.1277<br>0.1432|0.1383<br>**0.1519**|0.1300<br>0.1497|
|ConvNeXt-S<sub>(3)</sub><br>Skip-ConvNeXt-S<sub>(3)</sub>|0.1455<br>0.1625|0.1723<br>0.1839|0.1839<br>0.1965|0.1780<br>**0.1973**|

## Performances of models trained on CIFAR-10/100
Following models are trainied using the CIFAR-10/100 dataset from scratch. No extra images are used.

### Models
|Model|# of params|flops (32x32)|settings|
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
