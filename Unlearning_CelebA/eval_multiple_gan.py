import os
import sys
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from torchvision import transforms, utils
import classifier_models
# import gan_model
import torch
from torch import optim
# from vqvae import VQVAE
from collections import OrderedDict
from torchvision import utils
from model import Generator, Discriminator
# from stylegan2_ewc import mixing_noise
import torchvision.models as models
import torch.nn as nn
import wandb
# from fid import calc_fid,extract_feature_from_samples
import pandas as pd
import random
from calc_inception import load_patched_inception_v3

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

def copy_file_paths(folder_path, extension):
    file_paths = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    
    return file_paths

device='cuda'
resnext50_32x4d = models.resnext50_32x4d(pretrained=False)
resnext50_32x4d.fc = nn.Linear(2048, 40)
resnext50_32x4d.to(device)
path_toLoad="/classifier_celeba/classifier_github/model/model_3_epoch.pt"
checkpoint = torch.load(path_toLoad)
#Initializing the model with the model parameters of the checkpoint.
resnext50_32x4d.load_state_dict(checkpoint['model_state_dict'])
def img_classifier40(images,feature_type=None):
    global classifier_checkpoint
    device = "cuda"
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

        
    neg=converted_Score[15]
    neg_image = []
    pos_image = []

    
    
    
    

    
    neg_index = torch.where(neg == 1)[0]
    pos_index = torch.where(neg == 0)[0]
    # neg_index = torch.where(neg == 0)[0]
    # pos_index = torch.where(neg == 1)[0]
    # print(type(images),type(neg_index))
    neg_image.append(images[neg_index])
    pos_image.append(images[pos_index])

    neg_images = torch.cat(neg_image, dim=0)
    pos_images = torch.cat(pos_image, dim=0)
    
    # print(all_preds)
    return converted_Score,neg_images, pos_images,neg_index

def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

def norm_range(t, value_range):
    if value_range is not None:
        norm_ip(t, value_range[0], value_range[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))
        

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
        "--size", type=int, default=256, help="image sizes for generator"
    )
    # parser.add_argument(
    #     "--inception",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="path to precomputed inception embedding",
    # )
    # parser.add_argument(
    #     "--ckpt", metavar="CHECKPOINT", help="path to generator checkpoint"
    # )

    args = parser.parse_args()
    # wandb.init(project="Unlearning_Stylegan2",name="Evaluating GAN")

    neg_noise=[]
    initial_model = Generator(
                256, 512, 8, channel_multiplier=2,
            ).to(device)




    ckpt_gen="/stylegan2-pytorch/checkpoint/360000.pt"

    ckpt = torch.load(ckpt_gen)

    initial_model.load_state_dict(ckpt["g_ema"],strict=False)
    final_model = Generator(
                    256, 512, 8, channel_multiplier=2,
                ).to(device)



    result=[]
    fids=[]
    
    ##provide path to models to be evaluated
    ckpt_gens=["/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.0.pt",
               "/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.1.pt",
               "/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.2.pt",
               "/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-0.30000000000000004.pt",
               "/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.4.pt",
               "/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.5.pt",
               "/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-0.6000000000000001.pt",
               "/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-0.7000000000000001.pt",
               "/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.8.pt",
               "/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.9.pt",
               "/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-001.0.pt",
               "/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-001.1.pt",
               "/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-1.2000000000000002.pt",
               "/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-001.3.pt"
               

               
               
               ]

    # ckpt_gens=["/stylegan2-pytorch/checkpoints_noeyeglasses/390000.pt"]

    # folder_path = "/stylegan2-pytorch/checkpoints_nobangs"  # Replace with your folder path
    # extension = ".pt"
    # file_paths = copy_file_paths(folder_path, extension)

    # # Print the file paths
    # file_paths.sort()
    # # file_paths.reverse()
    # for path in file_paths:
    #     print(path)
    # result=[]
    # # fids=[]
    # ckpt_gens=file_paths
    # print(ckpt_gens)
    # ckpt_gens.reverse()
    cnt=0

    for ckpt_gen in ckpt_gens:
        # if(cnt%500!=0):
        #     cnt+=50
            
        #     continue

        # cnt+=50

        ckpt = torch.load(ckpt_gen)

        final_model.load_state_dict(ckpt)
        # sample_z=torch.load("/home/ece/hdd/Piyush/Unlearning-EBM/VQ-VAE/stylegan2/sample_z.pt")
            # for names,param in initial_model.named_parameters():
            #     param.data+=param_difference[names]*gamma
        latent_space=[]
        all_negs_old=[]
        all_negs_new=[]
        all_pos_old=[]
        all_pos_new=[]

        all_features_og=[0]*40
        all_features_new=[0]*40
        latent_space=[]
        latent_space=torch.load("/unlearning_gan/Unlearning_CelebA/latent_vector_Batch64.pt")
        # for i in range(78):
        #     latent_space.append(torch.randn(64,512).to(device))
        
        # latent_space=[sample_z]
        for ind in tqdm(range(len(latent_space))):
            gen_img = initial_model([latent_space[ind]])
            op=gen_img[0]
            norm_range(op,(-1,1))
            allnegs,neg, pos,negind = img_classifier40(op,)
            del gen_img
            del op
            allnegs,neg, pos = allnegs.detach().cpu(),neg.detach().cpu(), pos.detach().cpu()
            features_gan=[]
            

            for i in allnegs:
                
                

                features_gan.append(torch.sum(i).item())

            all_features_og=[sum(i) for i in zip(features_gan,all_features_og)]

            
            neg_noise.append(latent_space[ind][negind])
        
            all_negs_old.extend(neg)
            all_pos_old.extend(pos)


            


            
            gen_img = final_model([latent_space[ind]])
            op=gen_img[0]
            norm_range(op,(-1,1))
            allnegs,neg, pos,_ = img_classifier40(op,)
            del gen_img
            del op
            allnegs,neg, pos = allnegs.detach().cpu(),neg.detach().cpu(), pos.detach().cpu()
            features_gan=[]
            for i in allnegs:
                
                

                features_gan.append(torch.sum(i).item())

            all_features_new=[sum(i) for i in zip(features_gan,all_features_new)]
            
            
            # neg_noise.append(sample_z[ind][negind])
            all_negs_new.extend(neg)
            all_pos_new.extend(pos)


        if(len(all_negs_new)!=0):
            grid = utils.make_grid(all_negs_new[0:64], nrow=8, normalize=True, range=(0,1))
            # wandb.log({'Images_New_GAN': wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=ind)
            utils.save_image(
                            all_negs_new,
                            f"/unlearning_gan/Unlearning_CelebA/eval_images/{str(ind).zfill(6)}.png",
                            nrow=int(64 ** 0.5),
                            normalize=False,
                            range=(-1, 1),
                        )
        print("Images generated by old GANs:","Pos Images:",len(all_pos_old), "Neg Images:", len(all_negs_old))
        print("Images generated by new GANs:","Pos Images:",len(all_pos_new), "Neg Images:", len(all_negs_new))

        result.append((len(all_negs_old)-len(all_negs_new))/len(all_negs_old))
        # args = parser.parse_args()

        # ckpt = torch.load(ckpt_gen)

        # g = Generator(args.size, 512, 8).to(device)
        # g.load_state_dict(ckpt["g_ema"])
        # g = nn.DataParallel(g)
        # g.eval()

        # if args.truncation < 1:
        #     with torch.no_grad():
        #         mean_latent = g.mean_latent(args.truncation_mean)

        # else:
        #     mean_latent = None

        # inception = nn.DataParallel(load_patched_inception_v3()).to(device)
        # inception.eval()

        # features = extract_feature_from_samples(
        #     g, inception, args.truncation, mean_latent, args.batch, args.n_sample, device
        # ).numpy()
        # print(f"extracted {features.shape[0]} features")

        # sample_mean = np.mean(features, 0)
        # sample_cov = np.cov(features, rowvar=False)

        # with open(args.inception, "rb") as f:
        #     embeds = pickle.load(f)
        #     real_mean = embeds["mean"]
        #     real_cov = embeds["cov"]

        # fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
        # fids.append(fid)

    # print(fids)
    print(result)

    # df.to_csv(os.path.join(wandb.run.dir, "results_gan_Wearing_Hat_gamma_0.001_total_fixed.csv"))


