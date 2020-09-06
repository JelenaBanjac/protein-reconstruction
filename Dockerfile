ARG CUDA="10.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev \
 && apt-get install wget

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda \
 && rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/miniconda/bin:$PATH
RUN ls

# Create a Python 3.7 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name py37 python=3.7 \
 && /miniconda/bin/conda clean -ya \
 && /miniconda/bin/conda init

ENV CONDA_DEFAULT_ENV=py37
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN git clone https://github.com/JelenaBanjac/protein-reconstruction.git
RUN cd protein-reconstruction \
 && /miniconda/bin/conda env create -f environment.yml
RUN echo "source activate protein_reconstruction" > ~/.bashrc \
 && pip install tensorflow-gpu==2.0.0 \
 && pip install tensorflow-graphics-gpu

WORKDIR /protein-reconstruction

