python train.py --dataroot /data/jlim/cyclegan_data/keypress_data/ \
		--name keypress_cyclegan \
		--model cycle_gan \
		--dataset_mode semialigned \
		--direction BtoA \
		--gpu_ids 0,1,2,3 \
		--batch_size 16 


