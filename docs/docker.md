# Docker image with pytorch-CycleGAN-and-pix2pix

We provide both Dockerfile and pre-built Docker container that can run this code repo.

## Prerequisite

- Install [docker-ce](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- Install [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker#quickstart)

## Running with Dockerfile
We posted the [Dockerfile](Dockerfile). 

```bash
cd ~/
git clone https://github.com/1222-takeshi/pytorch-CycleGAN-and-pix2pix.git
cd pytorch-CycleGAN-and-pix2pix/docs
docker build -f DockerFile .
```

- Start an interactive docker session. `-p 8097:8097` option is needed if you want to run `visdom` server on the Docker container.

```bash
docker run -it -p 8097:8097 --gpus all <Docker ID>
```

- Now you are in the Docker environment. Go to our code repo and start running things.
```bash
bash datasets/download_pix2pix_dataset.sh facades
python -m visdom.server &
bash scripts/train_pix2pix.sh
```
