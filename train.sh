python train.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/train/ --name exp4 --dataset_mode adapterfull --model pix2pix --direction BtoA  --use_wandb --n_epochs 150  --output_nc 19 --epoch 123#--input_nc 19
#python train.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/train/ --name vcg_augmented_horiz_v2_500_epochs --dataset_mode vcg --model pix2pix --output_nc 19  --use_wandb --n_epochs 500 #--netG unet_256 --direction BtoA


#python train.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/train/ --name vcg --dataset_mode vcg --model pix2pix --input_nc 19 --output_nc 19 --direction BtoA --use_wandb --n_epochs 500

#python train.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/train/ --name vcg_changing_deconv_by_upsam --dataset_mode vcg --model pix2pix  --direction BtoA --use_wandb 
