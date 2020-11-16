#!/bin/bash
for i in {0..8000}
  do
    python cycleGAN_sketch_process.py --sketch_image "TestA/u${i}.png" --ground_image "TestB/u${i}.png" --output_path "/u${i}.png"
  done
