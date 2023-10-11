export CUDA_VISIBLE_DEVICES=0

python stylegan2_ewc.py --exp 8 --iter 2001 --g_ckpt /Stylegan2/200000.pt --size 32
