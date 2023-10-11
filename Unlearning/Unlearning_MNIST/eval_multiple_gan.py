import os
import sys

import pickle
import numpy as np
import argparse
from tqdm import tqdm
from fid import fid_func

# import gan_model
import torch
from torch import optim
# from vqvae import VQVAE
from collections import OrderedDict
from torchvision import utils
from model import Generator, Discriminator
from torchvision import transforms, utils
from Classifier_Mnist.lenet import LeNet5

import torchvision.models as models
import torch.nn as nn
import wandb
# from fid import calc_fid,extract_feature_from_samples
import pandas as pd
# from calc_inception import load_patched_inception_v3




device='cuda'
classifier_checkpoint = "Unlearning_MNIST/Classifier_Mnist/lenet_epoch=12_test_acc=0.991.pth"
def img_classifier(images,feature_type):
    # global classifier_checkpoint
    device = "cuda"
    # temp = []
    
    three_chan_to_1=transforms.Grayscale(num_output_channels=1)
    images_temp=three_chan_to_1(images)

    classifier = LeNet5().eval().to(device)
    ckpt=torch.load(classifier_checkpoint)
    classifier.load_state_dict(ckpt)

    preds=classifier(images_temp).cpu().detach().numpy()
    class_preds = np.argmax(preds, axis=1)
    # print(feature_type)
    neg_ind=[]
    pos_img=[]
    neg_img=[]
    cnt=0
    
    for preds in class_preds:
        if(preds==feature_type):
            neg_ind.append(cnt)
            # print("debug")
            neg_img.append(images[cnt].detach().cpu())
            # pos_img.append(0)

        else:
            # neg_img.append(0)
            pos_img.append(images[cnt].detach().cpu())

        cnt+=1

    # neg_ind = torch.cat(neg_ind, dim=0)
    # neg_img = torch.cat(neg_img, dim=0)
    # pos_img = torch.cat(pos_img, dim=0)
    
    return neg_img,pos_img,neg_ind

    
    
    
    

    
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
        

import os

def copy_file_paths(folder_path, extension):
    file_paths = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    
    return file_paths




if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Calculate FID scores")

    parser.add_argument("--truncation", type=float, default=1, help="truncation factor")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of samples to calculate mean for truncation",
    )
    parser.add_argument(
        "--batch", type=int, default=64, help="batch size for the generator"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=50000,
        help="number of the samples for calculating FID",
    )
    parser.add_argument(
        "--size", type=int, default=32, help="image sizes for generator"
    )
    # parser.add_argument(
    #     "--inception",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="path to precomputed inception embedding",
    # )
    parser.add_argument("--expt",type=int,default=8)
    # parser.add_argument(
    #     "--ckpt", metavar="CHECKPOINT", help="path to generator checkpoint"
    # )

    args = parser.parse_args()
    # wandb.init(project="Unlearning_Stylegan2",name="Evaluating GAN")
    args.latent=128
    neg_noise=[]
    initial_model = Generator(
                args.size, args.latent, 8, channel_multiplier=2,
            ).to(device)




    ##checkpoint for pretrained GAN
    ckpt_gen="Stylegan2/200000.pt"

    ckpt = torch.load(ckpt_gen)

    initial_model.load_state_dict(ckpt["g_ema"])
    final_model = Generator(
                    args.size, args.latent, 8, channel_multiplier=2,
                ).to(device)


    # Usage example
    folder_path = "Unlearning_MNIST/wandb/run-20230914_035736-q0u04va7/files"  # Replace with your folder path
    extension = ".pt"
    file_paths = copy_file_paths(folder_path, extension)

    # Print the file paths
    
    file_paths.sort()
    for path in file_paths:
        print(path)
    result=[]
    fids=[]
    ckpt_gens=file_paths
    ckpt_gens.sort()
    
    for ckpt_gen in ckpt_gens:
        ckpt = torch.load(ckpt_gen)

        final_model.load_state_dict(ckpt["g_ema"])

        
        latent_space=[]
        all_negs_old=[]
        all_negs_new=[]
        all_pos_old=[]
        all_pos_new=[]

        all_features_og=[0]*40
        all_features_new=[0]*40
        latent_space=[]
        ##latent vector for evaluation
        latent_space=torch.load("/Unlearning_MNIST/latent_vector_batch64.pt")
        # for i in range(78):
        #     latent_space.append(torch.randn(64,512).to(device))
        
        # latent_space=[sample_z]
        for ind in tqdm(range(len(latent_space))):
            latent_vector=latent_space[ind].to(device)

            gen_img = initial_model([latent_vector])
            op=gen_img[0]
            # norm_range(op,(-1,1))
            neg, pos,negind = img_classifier(op,args.expt)
            del gen_img
            del op
            
            

            

            
            neg_noise.append(latent_space[ind][negind])
        
            all_negs_old.extend(neg)
            all_pos_old.extend(pos)


            


            
            gen_img = final_model([latent_vector])
            op=gen_img[0]
            # norm_range(op,(-1,1))
            neg, pos,_ = img_classifier(op,args.expt)
            del gen_img
            del op
            # allnegs,neg, pos = allnegs.detach().cpu(),neg.detach().cpu(), pos.detach().cpu()
            
            
            
            # neg_noise.append(sample_z[ind][negind])
            all_negs_new.extend(neg)
            all_pos_new.extend(pos)



        print("Images generated by old GANs:","Pos Images:",len(all_pos_old), "Neg Images:", len(all_negs_old))
        print("Images generated by new GANs:","Pos Images:",len(all_pos_new), "Neg Images:", len(all_negs_new))

        result.append((len(all_negs_old)-len(all_negs_new))/len(all_negs_old))
        fid=fid_func(ckpt_gen,args.expt)
        fids.append(fid)
        print(fid)
        
        # args = parser.parse_args()

        
    print(result)
    print(fids)

    # df.to_csv(os.path.join(wandb.run.dir, "results_gan_Wearing_Hat_gamma_0.001_total_fixed.csv"))


