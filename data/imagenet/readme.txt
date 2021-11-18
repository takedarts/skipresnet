This is a directory of the ImageNet dataset.

[step 1] Download following files from ImageNet web site.
 - ILSVRC2012_devkit_t12.tar.gz
 - ILSVRC2012_img_train.tar
 - ILSVRC2012_img_val.tar

[step 2] Save these files in a dataset directory.
% tree data
data
└── imagenet
    ├── ILSVRC2012_devkit_t12.tar.gz
    ├── ILSVRC2012_img_train.tar
    ├── ILSVRC2012_img_val.tar
    └── readme.txt

[step 3] Run a training script `src/train.py`.
The ImageNet dataset is extracted and divided into shard files prior to the training process.
