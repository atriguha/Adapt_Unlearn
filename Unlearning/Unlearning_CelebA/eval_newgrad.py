import os
import sys


import argparse
from tqdm import tqdm
from torchvision import transforms
import classifier_models
# import gan_model
import torch
from torch import optim
# from vqvae import VQVAE
from collections import OrderedDict
from torchvision import utils
from model import Generator, Discriminator

import torchvision.models as models
import torch.nn as nn
import wandb
import pandas as pd
def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]



def img_classifier40(images,feature_type=None):
    device='cuda'
    resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
    resnext50_32x4d.fc = nn.Linear(2048, 40)
    resnext50_32x4d.to(device)
    path_toLoad="/classifier_celeba/classifier_github/model/model_3_epoch.pt"
    checkpoint = torch.load(path_toLoad)
    #Initializing the model with the model parameters of the checkpoint.
    resnext50_32x4d.load_state_dict(checkpoint['model_state_dict'])
    
    
    # temp = []

    with torch.no_grad():
          
        
        #Setting the model to be in evaluation mode. This will set the batch normalization parameters.
        resnext50_32x4d.eval() 
        res=transforms.Resize((218,178))
        images=res(images)
        # ip=torch.randn(8,3,218,178).to(device)
        scores=resnext50_32x4d(images)
        # labels=torch.zeros(8,40).to(device)
        converted_Score=scores.clone()
        converted_Score[converted_Score>=0]=1
        converted_Score[converted_Score<0]=0
        # print(converted_Score)
        converted_Score=converted_Score.t()


    all_preds = converted_Score
    feature_dict={'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, 'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, 'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}
    # print((feature_type))
    if(feature_type=="Beard"):
         
        ind=feature_dict["No_Beard"]

    else:
         ind=feature_dict[feature_type]
    # ind=feature_dict[feature_type]
    # print("feature index is:",ind)
    neg_image = []
    pos_image = []
    
    
    neg = all_preds[ind] #Index: 20; because we're suppresing Male
    if(feature_type=="Beard"):
         neg_index = torch.where(neg ==0)[0]
         pos_index = torch.where(neg == 1)[0]
    else:
             
        neg_index = torch.where(neg ==1)[0]
        pos_index = torch.where(neg == 0)[0]
    # pos_index = torch.where(neg ==1)[0]
    # neg_index = torch.where(neg == 0)[0]
    neg_image.append(images[neg_index])
    # neg_noise.append(noise[neg_index])
    # pos_noise.append(noise[pos])
    pos_image.append(images[pos_index])
    # neg_noise = torch.cat(neg_noise, dim=0)
    neg_images = torch.cat(neg_image, dim=0)
    pos_images = torch.cat(pos_image, dim=0)
    return all_preds,neg_images, pos_images,neg_index   
    # neg=converted_Score[5]
    

    
    
    
    

    
    # neg_index = torch.where(neg == 1)[0]
    # pos_index = torch.where(neg == 0)[0]
    # # print(type(images),type(neg_index))
    # neg_image.append(images[neg_index])
    # pos_image.append(images[pos_index])

    # neg_images = torch.cat(neg_image, dim=0)
    # pos_images = torch.cat(pos_image, dim=0)
    
    # # print(all_preds)
    # return converted_Score,neg_images, pos_images,neg_index

def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

def norm_range(t, value_range):
    if value_range is not None:
        norm_ip(t, value_range[0], value_range[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))
        
device='cuda'
def eval(final_model,feature_type):
    # wandb.init(project="Unlearning_Stylegan2",name="Evaluating GAN")
    initial_model = Generator(
                256, 512, 8, channel_multiplier=2,
            ).to(device)



    ##provide path to pre-trained GAN
    ckpt_gen="/stylegan2-pytorch/checkpoint/360000.pt"

    ckpt = torch.load(ckpt_gen)

    initial_model.load_state_dict(ckpt["g_ema"],strict=False)
    neg_noise=[]
    

    
    latent_space=[]
    all_negs_old=[]
    all_negs_new=[]
    all_pos_old=[]
    all_pos_new=[]
    all_img_new=[]
    # all_features_og=[0]*40
    # all_features_new=[0]*40
    latent_space=[]

    latent_space=torch.load("/unlearning_gan/Unlearning_CelebA/latent_vector_Batch64_eval.pt")
    # for i in range(78):
    #     latent_space.append(torch.randn(64,512).to(device))
    
    # latent_space=[sample_z]
    for ind in tqdm(range(len(latent_space))):
        gen_img = initial_model([latent_space[ind]])
        op=gen_img[0]
        norm_range(op,(-1,1))
        _,neg, pos,negind = img_classifier40(op,feature_type)
        
        del gen_img
        del op
        if(len(neg)==0):
             del neg,pos,negind
             continue
        
        neg, pos = neg.detach().cpu(), pos.detach().cpu()
        features_gan=[]
        

        

        
        neg_noise.append(latent_space[ind][negind])
    
        all_negs_old.extend(neg)
        all_pos_old.extend(pos)


        


        

        gen_img = final_model([latent_space[ind][negind]])
        op=gen_img[0]
        norm_range(op,(-1,1))
        
        _,neg, pos,_ = img_classifier40(op,feature_type)
        op=op.detach().cpu()
        all_img_new.extend(op)
        del gen_img
        del op
        neg, pos = neg.detach().cpu(), pos.detach().cpu()
        features_gan=[]
        all_negs_new.extend(neg)
        all_pos_new.extend(pos)


    

    return all_pos_old,all_pos_new,all_negs_new,all_negs_old,all_img_new
            


    
    


    print("Images generated by old GANs:","Pos Images:",len(all_pos_old), "Neg Images:", len(all_negs_old))
    print("Images generated by new GANs:","Pos Images:",len(all_pos_new), "Neg Images:", len(all_negs_new))

    




def main():
    initial_model = Generator(
                256, 512, 8, channel_multiplier=2,
            ).to(device)



    ##checkpoint of pre-trained GAN
    ckpt_gen="/checkpoint/actual_checkpoint.pt"

    ckpt = torch.load(ckpt_gen)

    initial_model.load_state_dict(ckpt["g_ema"],strict=False)
    final_model = Generator(
                    256, 512, 8, channel_multiplier=2,
                ).to(device)



    ##checkpoint of GAN to be evaluated
    ckpt_gen="/stylegan2/wandb/run-20230620_195635-d7fozq16/files/03000.pt"

    ckpt = torch.load(ckpt_gen)

    final_model.load_state_dict(ckpt["g_ema"],strict=False)

    eval(initial_model,final_model)
    # wandb.init(project="Unlearning_Stylegan2",name="Evaluating GAN")


if __name__ == "__main__":
    main()