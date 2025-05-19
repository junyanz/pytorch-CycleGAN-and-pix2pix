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
RUN git clone https://github.com/migueel-st/pix2pix-pokemon.git
WORKDIR /pix2pix-pokemon
RUN git checkout chore/setup

# Install dependencies
RUN conda env create -f environment.yml && conda clean -afy
ENV PATH=/opt/conda/envs/p2p/bin:$PATH
ENV CONDA_DEFAULT_ENV=p2p

# Setup nano for inline git commits
RUN apt-get update && apt-get install -y nano
RUN git config --global core.editor "nano"

# To show currently active git branch in terminal
RUN chmod +x ./.devcontainer/update-bashrc.sh
