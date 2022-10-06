#python train.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/train/ --name vcg --dataset_mode vcg --model pix2pix --input_nc 19 --output_nc 19 --direction BtoA --use_wandb 

python train.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/train/ --name vcg_changing_deconv_by_upsam --dataset_mode vcg --model pix2pix  --direction BtoA --use_wandb 
