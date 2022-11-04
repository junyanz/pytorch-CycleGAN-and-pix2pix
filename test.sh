python test.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/test --model pix2pix --name exp4_channels  --results_dir exp4_channels --dataset_mode adapter   --direction BtoA  --output_nc 4 --use_wandb


#python test.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix_CRC04/test --model pix2pix --name losses_1  --results_dir losses_1 --dataset_mode adapterfull  --output_nc 19 --direction BtoA 


#python test.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/test --model pix2pix --name exp4_case1  --results_dir exp4_case1 --dataset_mode adapterfull  --output_nc 18 --direction BtoA #--input_nc 3  # # --netG resnet_9blocks #


#python test.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/test --model pix2pix --name exp4_case1  --results_dir exp4_case1 --dataset_mode adapterfull  --output_nc 18 --direction BtoA #--input_nc 3  # # --netG resnet_9blocks #

#python test.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/ --direction BtoA --model pix2pix --name vcg_changing_deconv_by_upsam --use_wandb --results_dir borrar1