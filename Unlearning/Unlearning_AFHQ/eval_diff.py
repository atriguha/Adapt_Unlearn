##classifier link : https://github.com/rgkannan676/Recognition-and-Classification-of-Facial-Attributes/tree/main

import os
import sys
sys.path.append("/home/ece/hdd/Piyush/Unlearning-EBM")

import argparse
from tqdm import tqdm
import torchvision.models as models
import torch.nn as nn
import classifier_models
# import gan_model
import torch
from torch import optim
# from vqvae import VQVAE
from collections import OrderedDict
from torchvision import utils
from model import Generator, Discriminator



# @torch.no_grad()

# def get_checkpoint():
#     classifier_path="/home/ece/hdd/Piyush/Unlearning-EBM/classifier/checkpoints/resnet50_binary_second/checkpoint30.pt"
#     checkpoint = torch.load(
#         classifier_path)
#     checkpoint_ = {}

#     for key in checkpoint.keys():
#         new_key = key.replace("module.", "")
#         checkpoint_[new_key] = checkpoint[key]
#     return checkpoint_

device='cuda'
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
resnext50_32x4d.fc = nn.Linear(2048, 40)
resnext50_32x4d.to(device)
path_toLoad="/home/ece/hdd/Piyush/Stylegan2/classifier_celeba/classifier_github/model/model_3_epoch.pt"
checkpoint = torch.load(path_toLoad)
#Initializing the model with the model parameters of the checkpoint.
resnext50_32x4d.load_state_dict(checkpoint['model_state_dict'])
# classifier_checkpoint = get_checkpoint()
def img_classifier40(images,feature_type=None):
    global classifier_checkpoint
    device = "cuda"
    # temp = []

    with torch.no_grad():
          
        
        #Setting the model to be in evaluation mode. This will set the batch normalization parameters.
        resnext50_32x4d.eval() 
        # ip=torch.randn(8,3,218,178).to(device)
        scores=resnext50_32x4d(images)
        # labels=torch.zeros(8,40).to(device)
        converted_Score=scores.clone()
        converted_Score[converted_Score>=0]=1
        converted_Score[converted_Score<0]=0
        # print(converted_Score)
        converted_Score=converted_Score.t()

        
    neg=converted_Score[15]
    neg_image = []
    pos_image = []

    
    
    
    

    
    neg_index = torch.where(neg == 1)[0]
    pos_index = torch.where(neg == 0)[0]
    # print(type(images),type(neg_index))
    neg_image.append(images[neg_index])
    pos_image.append(images[pos_index])

    neg_images = torch.cat(neg_image, dim=0)
    pos_images = torch.cat(pos_image, dim=0)
    
    # print(all_preds)
    return converted_Score,neg_images, pos_images,neg_index



# def img_classifier40(images,feature_type=None):
#     global classifier_checkpoint
#     device = "cuda"
#     # temp = []

#     classifier = classifier_models.resnet50(False)
#     classifier.load_state_dict(classifier_checkpoint)

#     classifier = classifier.requires_grad_(False).to(device)
#     maxk = 1
#     neg_image = []
#     pos_image = []

#     top_k_preds = []
#     with torch.no_grad():

#         outputs = classifier(images)
#         for attr_scores in outputs:
#             # print(attr_scores)
#             _, attr_preds = attr_scores.topk(maxk, 1, True, True)
#             top_k_preds.append(attr_preds.t())
#     all_preds = torch.cat(top_k_preds, dim=0)
#     # {0: Bangs, 1: Beard, 2: Male, 3: High_Cheekbones, 4: Smiling, 5: Mustache}"
#     # print(len(all_preds))
#     neg = all_preds[0]  # Index: 24; because we're suppresing beard

#     if(feature_type=="beard2clean"):
#         neg_index = torch.where(neg == 0)[0]
#         pos_index = torch.where(neg == 1)[0]

#     else:
#         neg_index = torch.where(neg == 1)[0]
#         pos_index = torch.where(neg == 0)[0]
#     # print(type(images),type(neg_index))
#     neg_image.append(images[neg_index])
#     pos_image.append(images[pos_index])

#     neg_images = torch.cat(neg_image, dim=0)
#     pos_images = torch.cat(pos_image, dim=0)
    
#     # print(all_preds)
#     return all_preds,neg_images, pos_images,neg_index

feature_dict={'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, 'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, 'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

label_to_feature={}

for itr in feature_dict:
     label_to_feature[feature_dict[itr]]=itr
        


import pandas as pd
df = pd.DataFrame(columns=['Gamma','NegGAN','PosGAN','NegModGAN','PosModGAN','Difference','Percentage Blocking'])
df["Gamma"]=[""]*40
df["NegGAN"]=[""]*40
# df["NegGAN"][0]=len(neg_images)
df["PosGAN"]=[""]*40
# df["PosGAN"][0]=len(pos_images)
df["NegModGAN"]=[""]*40
# df["NegEBM"][0]=len(eneg_images)
df["PosModGAN"]=[""]*40
# df["PosEBM"][0]=len(epos_images)

df["Difference"]=[""]*40
df["Percentage Blocking"]=[""]*40
for itr in feature_dict:
     df[itr]=[""]*40



print(df)


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




gammas=[]
for i in range(20):
      gammas.append(i*(-0.1))

for i in range(11):
      gammas.append(i*(0.1))

# gammas=[-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0, 0.1, 0.2, 0.3, 0.4, 0.5 ,0.6, 0.7 ,0.8, 0.9 ,1.0]

print(gammas)
all_features=[]
cnt=0
flag=False
normalize=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

def norm_range(t, value_range):
    if value_range is not None:
        norm_ip(t, value_range[0], value_range[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
# param_difference=calc_difference_param()
ckpt_gens=[
     "/home/ece/hdd/Piyush/Stylegan2/stylegan2-pytorch/checkpoint/360000.pt",
     "/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-001.0.pt",
     "/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/results_paper/exp/run-20230829_000810-068mc48m/files/No_Eyeglasses_50000.0102000.pt",
     "/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/results_paper/1/(repel_loss)/run-20230907_011724-bq3mg7kx/files/No_Eyeglasses_500010.001500.pt",
     "/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/results_paper/repel_loss/run-20230807_095040-d0jxkntn/files/No_Eyeglasses_40001.001500.pt"
           ]
gamma=0
for ckpt in ckpt_gens:
    gamma+=1
    # if os.path.isdir(f"sample/weight/average"):
    #         pass
    # else:
    #         os.mkdir(f"sample/weight/average")
    initial_model = Generator(
                256, 512, 8, channel_multiplier=2,ckpt_disc=ckpt
            ).to(device)

    all_features_gamma=[0]*40

    cnt+=1


    # ckpt_gen="/home/ece/hdd/Piyush/Unlearning-EBM/VQ-VAE/stylegan2/checkpoint/actual_checkpoint.pt"

    ckpt = torch.load(ckpt)
    if(gamma==2):
         
        initial_model.load_state_dict(ckpt,strict=False) 

    if(gamma!=2):
         
        initial_model.load_state_dict(ckpt["g_ema"],strict=False)   
    
    
    # sample_z=[]
    sample_z=torch.load("/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/eval_hats_15000_batch64.pt")
    # latent_vector=torch.load("/home/ece/hdd/Piyush/Unlearning-EBM/VQ-VAE/stylegan2/sample_z.pt")
    # sample_z.append(latent_vector)
    # for names,param in initial_model.named_parameters():
    #     param.data+=param_difference[names]*gamma
    
    neg_images=[]
    pos_images=[]
    # allnegsgan=[0]*40
    # sample_z=sample_z[0:32]
        
    for ind in tqdm(range(len(sample_z))):
        # torch.manual_seed(i)
        # z_latent = mixing_noise(16, 512, 0.9, 'cuda')
        # print(sample_z[ind].shape)
        gen_img,_ = initial_model([sample_z[ind]])
        op=gen_img
        norm_range(op,(-1,1))
        allnegs,neg, pos,_ = img_classifier40(op,)
        del gen_img
        del op
        allnegs,neg, pos = allnegs.detach().cpu(),neg.detach().cpu(), pos.detach().cpu()
        # features_gan=[]
        

        # for i in allnegs:
            

            

        #     features_gan.append(torch.sum(i).item())
            
                
        #     # cnt+=1

        # allnegsgan=[sum(i) for i in zip(features_gan,allnegsgan)]
        # if os.path.isdir(f"eval_results/{gamma}"):
        #     pass
        # else:
        #     os.mkdir(f"eval_results/{gamma}")

        # if os.path.isdir(f"eval_results/{gamma}/pos"):
        #     pass
        # else:
        #     os.mkdir(f"eval_results/{gamma}/pos")

        # if(len(neg)!=0):
             
        #     utils.save_image(
        #                     neg,
        #                     f"eval_results/{gamma}/{str(ind).zfill(6)}.png",
        #                     nrow=int(64 ** 0.5),
        #                     normalize=False,
        #                     range=(-1, 1),
        #                 )
            

        # if(len(pos)!=0):
             
        #         utils.save_image(
        #                         pos,
        #                         f"eval_results/{gamma}/pos/{str(ind).zfill(6)}.png",
        #                         nrow=int(64 ** 0.5),
        #                         normalize=False,
        #                         range=(-1, 1),
        #                     )
        

        
        
        neg_images.extend(neg)
        pos_images.extend(pos)
        features_gan=[]
        

        for i in allnegs:
            
            

            features_gan.append(torch.sum(i).item())


        all_features_gamma=[sum(i) for i in zip(features_gan,all_features_gamma)]
        
    df["Gamma"][cnt]=gamma
    df["NegModGAN"][cnt]=len(neg_images)
    df["PosModGAN"][cnt]=len(pos_images)
    all_features.append(all_features_gamma)
    print(gamma)
    print(f"with gan: Neg- {len(neg_images)} | %: {len(neg_images)/(len(neg_images)+len(pos_images))} & Pos- {len(pos_images)} ")

    

    

    # with torch.no_grad():
    #                 initial_model.eval()
    #                 sample, _ = initial_model([sample_z])
    #                 utils.save_image(
    #                     sample,
    #                     f"sample/weight/average/{str(gamma)}.png",
    #                     nrow=int(64 ** 0.5),
    #                     normalize=False,
    #                     range=(-1, 1),
    #                 )
    
row=0



for itr in all_features:
    row+=1
    feat=0
    print(len(itr))
    if(row==1):
        for values in itr:
            df[label_to_feature[feat]][row]=values
            feat+=1
    
    else:
         for values in itr:
          df[label_to_feature[feat]][row]=values
        #   -df[label_to_feature[feat]][1]
          feat+=1
    
          

# print(df)
print(len(all_features))
print(len(all_features[0]))

df.to_csv("results_diff.csv")