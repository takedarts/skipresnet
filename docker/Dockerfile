FROM nvcr.io/nvidia/pytorch:22.03-py3

LABEL maintainer="Atsushi TAKEDA <takeda@cs.tohoku-gakuin.ac.jp>"

USER root
RUN apt-get update && \
    apt-get install -y libgl1-mesa-dev=21.2.6-0ubuntu0.1~20.04.2 && \
    apt-get clean
RUN pip install \
    opencv-python==4.5.5.64 \
    albumentations==1.1.0 \
    pytorch-lightning==1.5.10 \
    webdataset==0.1.103 \
    neptune-client==0.15.1 \
    wandb==0.12.11 \
    crcmod==1.7 \
    git+https://github.com/rwightman/pytorch-image-models@v0.1-weights-swinv2
RUN git clone -b v1.4.8 https://github.com/open-mmlab/mmcv && \
    cd mmcv && MMCV_WITH_OPS=1 FORCE_CUDA=1 python setup.py install && cd .. && \
    rm -rf mmcv 
RUN git clone -b v2.23.0 https://github.com/open-mmlab/mmdetection.git && \
    cd mmdetection && python setup.py install && cd .. && \
    rm -rf mmdetection
RUN git clone -b v0.23.0 https://github.com/open-mmlab/mmsegmentation && \
    cd mmsegmentation && python setup.py install && cd .. && \
    rm -rf mmsegmentation
