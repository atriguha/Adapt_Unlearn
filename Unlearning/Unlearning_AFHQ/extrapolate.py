from model import Generator, Discriminator
import torch
import matplotlib.pyplot as plt
import random 
import torch.autograd as autograd
from image_generator_util_ewc import FeedbackData
from torch.utils.data import DataLoader
from tqdm import tqdm

device='cuda'

import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm



def calc_difference_param():

    initial_model = Generator(
                256, 512, 8, channel_multiplier=2,
            ).to(device)




    ckpt_gen="/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/pretrained_checkpoint/200000.pt"

    ckpt = torch.load(ckpt_gen)

    initial_model.load_state_dict(ckpt["g_ema"],strict=False)
    # Define your initial model and fine-tuned model
    ckpt_gens=["/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/wandb/run-20240422_030427-gswf6p61/files/EWCcelebaHQwild_0_05000.pt",
                        "/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/wandb/run-20240422_085112-62r82adw/files/EWCcelebaHQwild_1_05000.pt"
                        , "/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/wandb/run-20240422_144547-vak9av6g/files/EWCcelebaHQwild_2_05000.pt"
                        ,"/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/wandb/run-20240422_203457-oqsbp0qd/files/EWCcelebaHQwild_3_05000.pt",
                        "/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/wandb/run-20240423_022754-5kvoaor1/files/EWCcelebaHQwild_4_05000.pt",
                        
                        
                        
                        
                        ]
    
    # start=5000

    # for i in range (1):
          
    #       ckpt_gens.append("/home/ece/hdd/Piyush/Unlearning-EBM/VQ-VAE/stylegan2/checkpoint/neg/500000000.0/50/00"+str(start)+".pt")
    #       start+=1000






    

    parameter_difference = {}
    for ckpt_gen in ckpt_gens:
        print(ckpt_gen)
        fine_tuned_model = Generator(
                256, 512, 8, channel_multiplier=2,ckpt_disc=None
            ).to(device)
          


        # ckpt_gen="/home/ece/hdd/Piyush/Unlearning-EBM/VQ-VAE/stylegan2/checkpoint/neg/500000000.0/50/055000.pt"

        ckpt = torch.load(ckpt_gen)

        fine_tuned_model.load_state_dict(ckpt["g_ema"])

        # Assume both models have the same number of parameters and identical shapes
        num_params = sum(p.numel() for p in initial_model.parameters())
        assert num_params == sum(p.numel() for p in fine_tuned_model.parameters())

        # Subtract the parameters of the fine-tuned model from the initial model
        
        cnt=0

        with torch.no_grad():
            for initial_param ,fine_tuned_param in zip(initial_model.named_parameters(), fine_tuned_model.named_parameters()):
                initial_param_name,initial_weight=initial_param
                fine_tuned_name,fine_tuned_weight=fine_tuned_param
                if(initial_param_name==fine_tuned_name):
                    cnt+=1
                difference = fine_tuned_weight - initial_weight
                a=(difference)
                
                try:
                    parameter_difference[initial_param_name]+=a
                except KeyError:
                    parameter_difference[initial_param_name]=a

    for itr in parameter_difference:
          value=parameter_difference[itr]
          avg_value=value/len(ckpt_gens)
          parameter_difference[itr]=avg_value

    # print(parameter_difference)
    return parameter_difference
gammas=[]
for i in range(20):
      gammas.append(i*(-0.1))

for i in range(11):
      gammas.append(i*(0.1))

# gammas=[-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0, 0.1, 0.2, 0.3, 0.4, 0.5 ,0.6, 0.7 ,0.8, 0.9 ,1.0]

print(gammas)

cnt=0

param_difference=calc_difference_param()
for gamma in gammas:
    
    if os.path.isdir(f"/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/sample/Wild"):
            pass
    else:
            os.mkdir(f"/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/sample/Wild")

    if os.path.isdir(f"/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/checkpoints/Wild"):
            pass
    else:
            os.mkdir(f"/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/checkpoints/Wild")


    if os.path.isdir(f"/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/sample/Wild/Unlearning"):
            pass
    else:
            os.mkdir(f"/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/sample/Wild/Unlearning")

    if os.path.isdir(f"/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/checkpoints/Wild/Unlearning"):
            pass
    else:
            os.mkdir(f"/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/checkpoints/Wild/Unlearning")
    initial_model = Generator(
                256, 512, 8, channel_multiplier=2,ckpt_disc="/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_CELEBA/pretrained_GAN_ckpt/360000.pt"
            ).to(device)


    cnt+=1


    ckpt_gen="/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/pretrained_checkpoint/200000.pt"

    ckpt = torch.load(ckpt_gen)
    
    initial_model.load_state_dict(ckpt["g_ema"],strict=False)

    sample_z=torch.load("/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/sample_z.pt")
    for names,param in initial_model.named_parameters():
        param.data+=param_difference[names]*gamma

    
    
    with torch.no_grad():
                    initial_model.eval()
                    sample, _ = initial_model([sample_z])
                    utils.save_image(
                        sample,
                        f"/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/sample/Wild/Unlearning/{str(gamma)}.png",
                        nrow=int(64 ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
    

    torch.save(initial_model.state_dict(),
                    f"/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/checkpoints/Wild/Unlearning/{str(gamma).zfill(6)}.pt",
                )



