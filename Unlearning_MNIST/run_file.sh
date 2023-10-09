export CUDA_VISIBLE_DEVICES=1

# python train_different_losses.py --expt 1\
#  --list_of_models /home/ece/hdd/Piyush/Stylegan2/unlearning_gan/EWC_ADAPT_MNIST/checkpoints/adapt_1/run-20230712_163409-thbf0su5/files/MODEL_MNIST0_1_02000.pt \
#      /home/ece/hdd/Piyush/Stylegan2/unlearning_gan/EWC_ADAPT_MNIST/checkpoints/adapt_1/run-20230712_164504-ae3nepq1/files/MODEL_MNIST1_1_02000.pt \
#      /home/ece/hdd/Piyush/Stylegan2/unlearning_gan/EWC_ADAPT_MNIST/checkpoints/adapt_1/run-20230712_165437-78j6dapg/files/MODEL_MNIST2_1_02000.pt \
#      /home/ece/hdd/Piyush/Stylegan2/unlearning_gan/EWC_ADAPT_MNIST/checkpoints/adapt_1/run-20230712_170451-ttfotsez/files/MODEL_MNIST3_1_02000.pt\
#      /home/ece/hdd/Piyush/Stylegan2/unlearning_gan/EWC_ADAPT_MNIST/checkpoints/adapt_1/run-20230712_171734-yyroas4u/files/MODEL_MNIST4_1_02000.pt\
#     --size 32   --ckpt /home/ece/hdd/Piyush/Stylegan2/200000.pt\
#     --iter 1001 --repel  --gamma 1 
# python fid.py --size 32 --ckpt /home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_MNIST/wandb/run-20230710_173447-8oymc6uv/files/No_Bangs100600.pt --path /home/ece/hdd/Piyush/Unlearning-EBM/MNIST/raw/MNIST

python eval_multiple_gan.py --expt 1