#losses branch to test loss plots
python train.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix_CRC04/train/ --name losses_1 --dataset_mode adapterfull --model pix2pix --direction BtoA  --use_wandb --n_epochs 300 --n_epochs_decay 1 --output_nc 19

#python train.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/train/ --name losses_1 --dataset_mode adapterfull1 --model pix2pix --direction BtoA  --use_wandb --n_epochs 5

#python train.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/train/ --name exp4_case1 --dataset_mode adapterfull --model pix2pix --direction BtoA  --use_wandb --n_epochs 250  --output_nc 18 

# python test.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/test --model pix2pix --name exp4_case1  --results_dir exp4_case1 --dataset_mode adapterfull  --direction BtoA  --output_nc 18 --direction BtoA

#python train.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/train/ --name vcg_augmented_horiz_v2_500_epochs --dataset_mode vcg --model pix2pix --output_nc 19  --use_wandb --n_epochs 500 #--netG unet_256 --direction BtoA


#python train.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/train/ --name vcg --dataset_mode vcg --model pix2pix --input_nc 19 --output_nc 19 --direction BtoA --use_wandb --n_epochs 500

#python train.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/train/ --name vcg_changing_deconv_by_upsam --dataset_mode vcg --model pix2pix  --direction BtoA --use_wandb 
