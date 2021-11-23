#!/bin/bash

SHAPE_SOURCE=/app/frontend/cyclegan_shapes
RESULT_DIR=/app/frontend/cyclegan_images
PROGRESS_LOG=/app/frontend/generator_state.py

# test one shape on all gans
for i in {5..105..5}; do
    cp checkpoints/oriflame_cyclegan_1/"$i"_net_G_A.pth checkpoints/oriflame_cyclegan_1/latest_net_G.pth
    python test.py  --gpu_ids -1 --dataroot $SHAPE_SOURCE --name oriflame_cyclegan_1 --model test --no_dropout --results_dir $RESULT_DIR/"$i"
    echo "generator_state = \"done gen 1 $i\"" > $PROGRESS_LOG
done

for i in {5..115..5}; do
  cp checkpoints/oriflame_cyclegan_2/"$i"_net_G_A.pth checkpoints/oriflame_cyclegan_2/latest_net_G.pth
  python test.py  --gpu_ids -1 --dataroot $SHAPE_SOURCE --name oriflame_cyclegan_2 --model test --no_dropout --results_dir $RESULT_DIR/"$i"
  echo "generator_state = \"done gen 2 $i\"" > $PROGRESS_LOG
done

for i in {5..115..5}; do
  cp checkpoints/oriflame_cyclegan_3/"$i"_net_G_A.pth checkpoints/oriflame_cyclegan_3/latest_net_G.pth
  python test.py  --gpu_ids -1 --dataroot $SHAPE_SOURCE --name oriflame_cyclegan_3 --model test --no_dropout --results_dir $RESULT_DIR/"$i"
  echo "generator_state = \"done gen 3 $i\"" > $PROGRESS_LOG
done

echo "generator_state = \"done\"" > $PROGRESS_LOG