FILE=$1

echo "Note: available models are edges2shoes, sat2map, and facades_label2photo"

echo "Specified [$FILE]"

mkdir -p ./checkpoints/${FILE}_pretrained
MODEL_FILE=./checkpoints/${FILE}_pretrained/latest_net_G.pth
URL=https://people.eecs.berkeley.edu/~taesung_park/pytorch-CycleGAN-and-pix2pix/pix2pix_models/$FILE.pth

wget -N $URL -O $MODEL_FILE


