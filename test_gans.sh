#!/bin/bash

source /home/jakub/github/bp/pytorch-CycleGAN-and-pix2pix_2/env/bin/activate


# from original data generate originals
#for i in {5..105..5}; do
#    cp checkpoints/oriflame_cyclegan_1/"$i"_net_G_A.pth checkpoints/oriflame_cyclegan_1/latest_net_G.pth
#    python test.py  --gpu_ids -1 --dataroot datasets/oriflame_1/trainA --name oriflame_cyclegan_1 --model test --no_dropout --results_dir ./results2/"$i"
#done
#
#for i in {5..115..5}; do
#  cp checkpoints/oriflame_cyclegan_2/"$i"_net_G_A.pth checkpoints/oriflame_cyclegan_2/latest_net_G.pth
#  python test.py  --gpu_ids -1 --dataroot datasets/oriflame_2/trainA --name oriflame_cyclegan_2 --model test --no_dropout --results_dir ./results2/"$i"
#done
#
#for i in {5..115..5}; do
#  cp checkpoints/oriflame_cyclegan_3/"$i"_net_G_A.pth checkpoints/oriflame_cyclegan_3/latest_net_G.pth
#  python test.py  --gpu_ids -1 --dataroot datasets/oriflame_3/testA --name oriflame_cyclegan_3 --model test --no_dropout --results_dir ./results2/"$i"
#done


# test one shape
#rsync -a checkpoints/oriflame_cyclegan_2/95_net_G_A.pth one_off_tests/oriflame_cyclegan_2_95/latest_net_G.pth
#python test.py  --gpu_ids -1 --dataroot one_off_tests/dataset/shape --name oriflame_cyclegan_2 --model test --no_dropout --results_dir ./one_off_tests/results


# test one shape on all gans
for i in {5..105..5}; do
    cp checkpoints/oriflame_cyclegan_1/"$i"_net_G_A.pth checkpoints/oriflame_cyclegan_1/latest_net_G.pth
    python test.py  --gpu_ids -1 --dataroot one_off_tests/dataset/shape --name oriflame_cyclegan_1 --model test --no_dropout --results_dir /tmp/gan_results/"$i"
done

for i in {5..115..5}; do
  cp checkpoints/oriflame_cyclegan_2/"$i"_net_G_A.pth checkpoints/oriflame_cyclegan_2/latest_net_G.pth
  python test.py  --gpu_ids -1 --dataroot one_off_tests/dataset/shape --name oriflame_cyclegan_2 --model test --no_dropout --results_dir /tmp/gan_results/"$i"
done

for i in {5..115..5}; do
  cp checkpoints/oriflame_cyclegan_3/"$i"_net_G_A.pth checkpoints/oriflame_cyclegan_3/latest_net_G.pth
  python test.py  --gpu_ids -1 --dataroot one_off_tests/dataset/shape --name oriflame_cyclegan_3 --model test --no_dropout --results_dir /tmp/gan_results/"$i"
done