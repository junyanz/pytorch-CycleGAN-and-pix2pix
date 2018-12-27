# Docker image with pytorch-CycleGAN-and-pix2pix

We provide both Dockerfile and pre-built Docker container that can run this code repo.

## Prerequisite

- Install [docker-ce](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quickstart)

## Running pre-built Dockerfile

- Pull the pre-built docker file

```bash
docker pull taesungp/pytorch-cyclegan-and-pix2pix
```

- Start an interactive docker session. `-p 8097:8097` option is needed if you want to run `visdom` server on the Docker container.

```bash
nvidia-docker run -it -p 8097:8097  taesungp/pytorch-cyclegan-and-pix2pix
```

- Now you are in the Docker environment. Go to our code repo and start running things.
```bash
cd /workspace/pytorch-CycleGAN-and-pix2pix
bash datasets/download_pix2pix_dataset.sh facades
python -m visdom.server &
bash scripts/train_pix2pix.sh
```

## Running with Dockerfile

We also posted the [Dockerfile](Dockerfile). To generate the pre-built file, download the Dockerfile in this directory and run
```bash
docker build -t [target_tag] .
```
in the directory that contains the Dockerfile.
