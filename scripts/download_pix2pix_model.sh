FILE=$1

echo "Note: available models are edges2shoes, sat2map, map2sat, facades_label2photo, and day2night"
echo "Specified [$FILE]"

ROOT_TARGET_DIR="/home/user/data/phyusformer_data/post_miccai_exps/pix2pix_checkpoints"
# mkdir -p $ROOT_TARGET_DIR
TARGET_DIR=$ROOT_TARGET_DIR/$FILE/
mkdir -p $TARGET_DIR
MODEL_FILE=$TARGET_DIR/latest_net_G.pth
URL=http://efrosgans.eecs.berkeley.edu/pix2pix/models-pytorch/$FILE.pth

wget -N $URL -O $MODEL_FILE
