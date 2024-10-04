FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# install conda
RUN apt-get update && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh \
    && /bin/bash Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniconda3-py310_23.5.2-0-Linux-x86_64.sh

ENV PATH /opt/conda/bin:$PATH

# install conda update and build tools
RUN conda update -y -q conda \
    && conda install -y -q conda-build

# install python library
# install faiss-gpu
RUN conda install -y -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl \
    && conda install -c conda-forge openjdk=11

# install pytorch for cuda 11.4
# RUN conda install -y -c pytorch -c nvidia pytorch pytorch-cuda=11.4
RUN pip install torch

# install requirements
# このrequirements.txtの中身はpyseriniのみでok
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# set workdir
WORKDIR /workspace