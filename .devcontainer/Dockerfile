FROM nvidia/cuda:10.1-base

#Nvidia Public GPG Key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt update && apt install -y wget unzip curl bzip2 git
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN conda install -y pytorch torchvision -c pytorch
RUN mkdir /workspace/ && cd /workspace/
RUN git clone https://github.com/migueel-st/pix2pix-pokemon.git
RUN git checkout chore/setup
RUN cd pytorch-CycleGAN-and-pix2pix
RUN pip install -r requirements.txt

WORKDIR /workspace
