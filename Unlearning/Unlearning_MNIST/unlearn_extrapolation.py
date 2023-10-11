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



    ##checkpoint of initial model
    ckpt_gen="/stylegan2/checkpoint/actual_checkpoint.pt"

    ckpt = torch.load(ckpt_gen)

    initial_model.load_state_dict(ckpt["g_ema"],strict=False)
    # checkpoints of fine-tuned models
    ckpt_gens=["/stylegan2/checkpoint/bangs/neg000000/500000000.0/5000/005000.pt",
               "/stylegan2/checkpoint/bangs/neg000001/500000000.0/5000/005000.pt"
               , "/stylegan2/checkpoint/bangs/neg000002/500000000.0/5000/005000.pt"
               ,"/stylegan2/checkpoint/bangs/neg000003/500000000.0/5000/005000.pt",
               "/stylegan2/checkpoint/bangs/neg000004/500000000.0/5000/005000.pt"
               
               
               ]
    # ckpt_gens=["/stylegan2/checkpoint/bangs/neg/500000000.0/50/005000.pt"
               
            #    ]
    # start=5000

    # for i in range (1):
          
    #       ckpt_gens.append("/stylegan2/checkpoint/neg/500000000.0/50/00"+str(start)+".pt")
    #       start+=1000






    

    parameter_difference = {}
    for ckpt_gen in ckpt_gens:
        print(ckpt_gen)
        fine_tuned_model = Generator(
                256, 512, 8, channel_multiplier=2,ckpt_disc=None
            ).to(device)
          


        # ckpt_gen="/stylegan2/checkpoint/neg/500000000.0/50/055000.pt"

        ckpt = torch.load(ckpt_gen)

        fine_tuned_model.load_state_dict(ckpt["g_ema"],strict=False)

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
    if os.path.isdir(f"sample/Bangs"):
            pass
    else:
            os.mkdir(f"sample/Bangs")

    if os.path.isdir(f"checkpoint/Bangs"):
            pass
    else:
            os.mkdir(f"checkpoint/Bangs")


    if os.path.isdir(f"sample/Bangs/Unlearning"):
            pass
    else:
            os.mkdir(f"sample/Bangs/Unlearning")

    if os.path.isdir(f"checkpoint/Bangs/Unlearning"):
            pass
    else:
            os.mkdir(f"checkpoint/Bangs/Unlearning")
    initial_model = Generator(
                256, 512, 8, channel_multiplier=2,ckpt_disc="/stylegan2/checkpoint/actual_checkpoint.pt"
            ).to(device)


    cnt+=1


    ckpt_gen="/stylegan2/checkpoint/actual_checkpoint.pt"

    ckpt = torch.load(ckpt_gen)
    
    initial_model.load_state_dict(ckpt["g_ema"],strict=False)

    sample_z=torch.load("/stylegan2/sample_z.pt")
    for names,param in initial_model.named_parameters():
        param.data+=param_difference[names]*gamma

    
    
    with torch.no_grad():
                    initial_model.eval()
                    sample, _ = initial_model([sample_z])
                    utils.save_image(
                        sample,
                        f"sample/Bangs/Unlearning/{str(gamma)}.png",
                        nrow=int(64 ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
    

    torch.save(initial_model.state_dict(),
                    f"checkpoint/Bangs/Unlearning/{str(gamma).zfill(6)}.pt",
                )



