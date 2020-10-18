python train.py --dataroot /storage/easle --name artgen \
--preprocess='resize_and_crop' --gpu_ids=0 \
--model cycle_gan --no_dropout \
--update_html_freq 1 --display_freq 1