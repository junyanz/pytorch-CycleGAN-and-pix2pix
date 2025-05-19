FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

#Nvidia Public GPG Key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt update && apt install -y wget unzip curl bzip2 git
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

# Clone the project
WORKDIR /workspace
RUN git clone https://github.com/migueel-st/pix2pix-pokemon.git
WORKDIR /workspace/pix2pix-pokemon
RUN git checkout chore/setup

# Install dependencies
RUN conda env create -f environment.yml && conda clean -afy
ENV PATH=/opt/conda/envs/pytorch-CycleGAN-and-pix2pix/bin:$PATH
ENV CONDA_DEFAULT_ENV=pytorch-CycleGAN-and-pix2pix
