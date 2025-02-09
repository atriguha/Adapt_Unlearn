export CUDA_VISIBLE_DEVICES=1

# python extrapolate.py
# python unlearn_main.py --gamma 100 --iter 3000 --exp wild --ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/pretrained_checkpoint/200000.pt  --repel_loss --wandb

# python unlearn_main.py --gamma 1 --iter 3000 --exp dog --ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/pretrained_checkpoint/200000.pt  --repel_loss --wandb

# python unlearn_main.py --gamma 10 --iter 3000 --exp cat --ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/pretrained_checkpoint/200000.pt  --repel_loss --wandb

# python unlearn_main.py --gamma 0.1 --iter 3000 --exp wild --ckpt /home/ece/hdd/Piyush/Stylegan2/stylegan2-affhq/checkpoints_affhq/200000.pt --repel_loss --wandb



# python eval_multiple_gan.py --feature cat --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/ewc/wild/offline-run-20240425_230346-87lmmwjk

# python unlearn_ewc.py --ckpt /home/ece/hdd/Piyush/Stylegan2/stylegan2-pytorch/checkpoint/360000.pt  --iter 2001 --exp Bald --wandb \

# python unlearn_ewc.py --ckpt /home/ece/hdd/Piyush/Stylegan2/stylegan2-pytorch/checkpoint/360000.pt  --iter 2001 --exp Bald --wandb \

        
# python unlearn_ewc.py --ckpt /home/ece/hdd/Piyush/Stylegan2/stylegan2-pytorch/checkpoint/360000.pt  --iter 2001 --exp Bangs --wandb \
        

# python l2_unlearn.py --ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/pretrained_checkpoint/200000.pt --gamma 10 --iter 6000 --exp dog --wandb \

# python l2_unlearn.py --ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/pretrained_checkpoint/200000.pt --gamma 10 --iter 6000 --exp wild --wandb \

# python l2_unlearn.py --ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/pretrained_checkpoint/200000.pt --gamma 10 --iter 6000 --exp cat --wandb \

##extrapolation
# python fid.py --feature cat --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/CDC/cat
 
# python eval_multiple_gan.py --feature cat --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/RSSA/cat2nocat_self_dis_proj_10

python ret_fid.py --feature cat --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/orginal_samples/cat_expo --retrain_ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/stylegan2-affhq/checkpoints_no_cat/110000.pt


# python fid.py --feature dog --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/CDC/dog
 
# python eval_multiple_gan.py --feature dog --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/RSSA/dog2nodog_self_dis_proj_10

# python ret_fid.py --feature dog --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/CDC/dog --retrain_ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/stylegan2-affhq/checkpoints_no_dog/100000.pt


# python fid.py --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/CDC/wild
 
# python eval_multiple_gan.py --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/RSSA/wild2nowild_self_dis_proj_10

# python ret_fid.py --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/CDC/wild --retrain_ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/stylegan2-affhq/checkpoints_no_wild/220000.pt


##retrain
# python fid.py --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/repel/wild/offline-run-20240427_012406-vpddbeav
 
# python eval_multiple_gan.py --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/repel/wild/offline-run-20240427_012406-vpddbeav


## REPEL

# python ret_fid.py --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/repel/wild/offline-run-20240427_012406-vpddbeav --retrain_ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/stylegan2-affhq/checkpoints_no_wild/220000.pt


# python fid.py --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/repel/wild/offline-run-20240427_012406-vpddbeav
 
# python eval_multiple_gan.py --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/repel/wild/offline-run-20240427_012406-vpddbeav

## EXPO

# python ret_fid.py --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/expo/wild/offline-run-20240429_110530-osyf68m1 --retrain_ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/stylegan2-affhq/checkpoints_no_wild/220000.pt


# python fid.py --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/expo/wild/offline-run-20240429_110530-osyf68m1

# python eval_multiple_gan.py --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/expo/wild/offline-run-20240429_110530-osyf68m1

# ##CAT RECI
# python ret_fid.py --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/reci/wild/offline-run-20240429_015224-4yle6u39 --retrain_ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/stylegan2-affhq/checkpoints_no_wild/220000.pt


# python fid.py --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/reci/wild/offline-run-20240429_015224-4yle6u39

# python eval_multiple_gan.py --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/reci/wild/offline-run-20240429_015224-4yle6u39

# ##CAT EWC
# python ret_fid.py --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/ewc/wild/offline-run-20240425_230346-87lmmwjk --retrain_ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/stylegan2-affhq/checkpoints_no_wild/220000.pt

# python fid.py  --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/ewc/wild/offline-run-20240425_230346-87lmmwjk

# python eval_multiple_gan.py  --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/ewc/wild/offline-run-20240425_230346-87lmmwjk

# ##CAT EFK

# python ret_fid.py --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/efk/wild/offline-run-20240423_035832-lirzalur --retrain_ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/stylegan2-affhq/checkpoints_no_wild/220000.pt


# python fid.py  --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/efk/wild/offline-run-20240423_035832-lirzalur

# python eval_multiple_gan.py  --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/efk/wild/offline-run-20240423_035832-lirzalur

# ##CAT CFK
# python ret_fid.py --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/cfk/wild/offline-run-20240423_005849-po0d2n31 --retrain_ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/stylegan2-affhq/checkpoints_no_wild/220000.pt

# python fid.py  --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/cfk/wild/offline-run-20240423_005849-po0d2n31

# python eval_multiple_gan.py  --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/cfk/wild/offline-run-20240423_005849-po0d2n31


# ##CAT l1
# python ret_fid.py --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/l1/wild/offline-run-20240422_212317-qmhvvaoz --retrain_ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/stylegan2-affhq/checkpoints_no_wild/220000.pt

# python fid.py  --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/l1/wild/offline-run-20240422_212317-qmhvvaoz

# python eval_multiple_gan.py  --feature wild --folder_path /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/results_paper/l1/wild/offline-run-20240422_212317-qmhvvaoz



# python ret_fid.py --feature Bald --folder_path /home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/adapt_CDC/checkpoints/Bald --retrain_ckpt /home/ece/hdd/Piyush/Stylegan2/stylegan2-pytorch/checkpoints_nobald/310000.pt


# python ret_fid.py --feature Wearing_Hat --folder_path /home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/adapt_CDC/checkpoints/Wearing_Hat/final_ckpt --retrain_ckpt /home/ece/hdd/Piyush/Stylegan2/stylegan2-pytorch/checkpoints_nohats/410000.pt

# python ret_fid.py --feature Bangs --folder_path /home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/adapt_CDC/checkpoints/Bangs/final_ckpt --retrain_ckpt /home/ece/hdd/Piyush/Stylegan2/stylegan2-pytorch/checkpoints_nobangs/360000.pt

# python ret_fid.py --feature Bald --folder_path /home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/adapt_CDC/checkpoints/Bald/final_ckpt --retrain_ckpt /home/ece/hdd/Piyush/Stylegan2/stylegan2-pytorch/checkpoints_nobald/310000.pt

# python ret_fid.py --feature Eyeglasses --folder_path /home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/adapt_CDC/checkpoints/Eyeglasses/final_ckpt --retrain_ckpt /home/ece/hdd/Piyush/Stylegan2/stylegan2-pytorch/checkpoints_noeyeglasses/380000.pt





# python adapt_cfk.py --ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/pretrained_checkpoint/200000.pt  --iter 6000 --exp dog --wandb \

# python adapt_cfk.py --ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/pretrained_checkpoint/200000.pt  --iter 6000 --exp wild --wandb \

# python adapt_cfk.py --ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/pretrained_checkpoint/200000.pt  --iter 6000 --exp cat --wandb \


# python adapt_cfk.py --ckpt /home/ece/hdd/Piyush/Stylegan2/stylegan2-pytorch/checkpoint/360000.pt  --iter 6000 --exp Eyeglasses --wandb \

# python adapt_cfk.py --ckpt /home/ece/hdd/Piyush/Stylegan2/stylegan2-pytorch/checkpoint/360000.pt  --iter 6000 --exp Eyeglasses --wandb \

# python l2_unlearn.py --ckpt /home/ece/hdd/Piyush/Stylegan2/stylegan2-pytorch/checkpoint/360000.pt --gamma 10 --iter 6001 --exp Eyeglasses --wandb \

# python l2_unlearn.py --ckpt /home/ece/hdd/Piyush/Stylegan2/stylegan2-pytorch/checkpoint/360000.pt --gamma 10 --iter 6001 --exp Eyeglasses --wandb \

# python l2_unlearn.py --ckpt /home/ece/hdd/Piyush/Stylegan2/stylegan2-pytorch/checkpoint/360000.pt --gamma 10 --iter 6000 --exp Wearing_Hat --wandb \


# python unlearn_ewc.py --ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/pretrained_checkpoint/200000.pt  --iter 2001 --exp cat --wandb \

# python unlearn_ewc.py --ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/pretrained_checkpoint/200000.pt  --iter 2001 --exp dog --wandb \

# python unlearn_ewc.py --ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/pretrained_checkpoint/200000.pt  --iter 2001 --exp wild --wandb \

