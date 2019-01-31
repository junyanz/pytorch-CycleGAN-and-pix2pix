FILE=$1

echo "Note: available models are edges2shoes, sat2map, map2sat, facades_label2photo, and day2night"
echo "Specified [$FILE]"

mkdir -p ./checkpoints/${FILE}_pretrained
MODEL_FILE=./checkpoints/${FILE}_pretrained/latest_net_G.pth
URL=http://efrosgans.eecs.berkeley.edu/pix2pix/models-pytorch/$FILE.pth

wget -N $URL -O $MODEL_FILE
