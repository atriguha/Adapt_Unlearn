export CUDA_VISIBLE_DEVICES=1

# python train_different_losses.py --expt 1\
#  --list_of_models /unlearning_gan/EWC_ADAPT_MNIST/checkpoints/adapt_1/run-20230712_163409-thbf0su5/files/MODEL_MNIST0_1_02000.pt \
#      /unlearning_gan/EWC_ADAPT_MNIST/checkpoints/adapt_1/run-20230712_164504-ae3nepq1/files/MODEL_MNIST1_1_02000.pt \
#      /unlearning_gan/EWC_ADAPT_MNIST/checkpoints/adapt_1/run-20230712_165437-78j6dapg/files/MODEL_MNIST2_1_02000.pt \
#      /unlearning_gan/EWC_ADAPT_MNIST/checkpoints/adapt_1/run-20230712_170451-ttfotsez/files/MODEL_MNIST3_1_02000.pt\
#      /unlearning_gan/EWC_ADAPT_MNIST/checkpoints/adapt_1/run-20230712_171734-yyroas4u/files/MODEL_MNIST4_1_02000.pt\
#     --size 32   --ckpt /200000.pt\
#     --iter 1001 --repel  --gamma 1 --loss_type exp

# python eval_multiple_gan.py --expt 1