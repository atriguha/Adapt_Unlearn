import os
import sys
# sys.path.append("/home/ece/hdd/Piyush/Unlearning-EBM")
from classifier_models import CustomResNet50
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

device='cuda'

def get_checkpoint():
    # checkpoint = torch.load(
    # 			"/home/ece/hdd/Piyush/Unlearning-EBM/model_best40.pt")
    checkpoint=torch.load("/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/classifier_AFFHQ/resnet50/best_trained.pt",map_location=device)
    checkpoint_ = {}

    for key in checkpoint.keys():
        new_key = key.replace("module.","")
        checkpoint_[new_key] = checkpoint[key]
    return checkpoint_

classifier_checkpoint = get_checkpoint()
def img_classifier40(images,feature_type,images_org):
    global classifier_checkpoint
    device = "cuda"
    # temp = []
    images=images
    classifier = CustomResNet50(in_channels=3,num_classes=3)
    classifier.load_state_dict(classifier_checkpoint["model_state_dict"])

    classifier = classifier.requires_grad_(False).to(device)

    classifier.requires_grad_(False)
    maxk = 1
    neg_image = []
    pos_image = []
    # neg_noise = []
    # data = TensorDataset(images,noise)
    
    # print(data_loader.__dict__)
    # for data in data_loader:
    # 	data = data[0]
    top_k_preds = []
    with torch.no_grad():

        outputs = classifier(images) 

        # print(outputs.shape) 
        max_indices = torch.argmax(outputs, dim=1)

# Create a one-hot encoded tensor using torch.zeros_like()
        attr_preds = torch.zeros_like(outputs)

        # Set the element at the index of max_indices to 1
        attr_preds.scatter_(1, max_indices.view(-1, 1), 1)
        # print(attr_preds)
        top_k_preds=attr_preds.t()
        # print(top_k_preds.shape)
        # for attr_scores in outputs:
        #     # print(attr_scores)
        #     _, attr_preds = attr_scores.topk(maxk, 1, True, True)
        #     top_k_preds.append(attr_preds.t())
    # all_preds = torch.cat(top_k_preds, dim=0)
    all_preds=top_k_preds
    #{'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, 'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, 'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}
    feature_dict={'cat':0, 'dog':1, 'wild':2}
    # print(type(feature_type))
    # if(feature_type=="Beard"):
         
    #     ind=feature_dict["No_Beard"]

    # else:
    ind=feature_dict[feature_type]
    # print("feature index is:",ind)
    neg = all_preds[ind] #Index: 20; because we're suppresing Male
    
    # if(feature_type=="Beard"):
         
    #     pos_index = torch.where(neg ==1)[0]
    #     neg_index = torch.where(neg == 0)[0]

    # else:
    neg_index = torch.where(neg ==1)[0]
    pos_index = torch.where(neg == 0)[0]
    neg_image.append(images_org[neg_index])
    # neg_noise.append(noise[neg_index])
    # pos_noise.append(noise[pos])
    pos_image.append(images_org[pos_index])
    # neg_noise = torch.cat(neg_noise, dim=0)
    neg_images = torch.cat(neg_image, dim=0)
    pos_images = torch.cat(pos_image, dim=0)
    return all_preds,neg_images, pos_images,neg_index


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




    ckpt_gen="/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/pretrained_checkpoint/200000.pt"

    ckpt = torch.load(ckpt_gen)

    initial_model.load_state_dict(ckpt["g_ema"])
    neg_noise=[]
    

    # sample_z=torch.load("/home/ece/hdd/Piyush/Unlearning-EBM/VQ-VAE/stylegan2/sample_z.pt")
        # for names,param in initial_model.named_parameters():
        #     param.data+=param_difference[names]*gamma
    latent_space=[]
    all_negs_old=[]
    all_negs_new=[]
    all_pos_old=[]
    all_pos_new=[]
    all_img_new=[]
    # all_features_og=[0]*40
    # all_features_new=[0]*40
    latent_space=[]
    latent_space=torch.load("/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/Unlearning_AFFHQ/latent_vector_Batch64.pt")
    # for i in range(78):
    #     latent_space.append(torch.randn(64,512).to(device))
    
    # latent_space=[sample_z]
    for ind in tqdm(range(len(latent_space))):
        # print(latent_space[ind].shape)
        gen_img = initial_model([latent_space[ind]])
        op=gen_img[0]
        # norm_range(op,(-1,1))
        
        resize_transform=transforms.Resize((224,224))
        gen_img_resized=resize_transform(op)
        _,neg, pos,negind = img_classifier40(gen_img_resized,feature_type,op)
        # print("no of negative images found",len(negind))
        del gen_img,gen_img_resized
        del op
        # if(len(neg)==0):
        #      del neg,pos,negind
        #      continue
        
        neg, pos = neg.detach().cpu(), pos.detach().cpu()
        features_gan=[]
        

        

        
        neg_noise.append(latent_space[ind][negind])
    
        all_negs_old.extend(neg)
        all_pos_old.extend(pos)


        


        if(len(negind)!=0):
             

            gen_img = final_model([latent_space[ind][negind]])
            op=gen_img[0]
            # norm_range(op,(-1,1))
            gen_img_resized=resize_transform(op)
            
            _,neg, pos,_ = img_classifier40(gen_img_resized,feature_type,op)
            op=op.detach().cpu()
            all_img_new.extend(op)
            del gen_img,gen_img_resized
            del op
            neg, pos = neg.detach().cpu(), pos.detach().cpu()
            features_gan=[]
            all_negs_new.extend(neg)
            all_pos_new.extend(pos)


    # if(len(all_negs_old)!=0):
    #     grid = utils.make_grid(all_negs_old[0:64], nrow=8, normalize=True, range=(0,1))
    #     wandb.log({'Images_old_GAN': wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=ind)
         


    # if(len(all_img_new)!=0):
    #         grid = utils.make_grid(all_img_new[0:64], nrow=8, normalize=True, range=(0,1))
    #         wandb.log({'Images_New_GAN': wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=ind)
            # utils.save_image(
            #                 neg,
            #                 f"/home/ece/hdd/Piyush/Unlearning-EBM/VQ-VAE/stylegan2/eval_results/gan_newgrad/trained_gan/neg/{str(ind).zfill(6)}.png",
            #                 nrow=int(64 ** 0.5),
            #                 normalize=False,
            #                 range=(-1, 1),
            #             )
            
    
    print("Images generated by old GANs:","Pos Images:",len(all_pos_old), "Neg Images:", len(all_negs_old))
    print("Images generated by new GANs:","Pos Images:",len(all_pos_new), "Neg Images:", len(all_negs_new))
    return all_pos_old,all_pos_new,all_negs_new,all_negs_old,all_img_new
            


    
    


    

    




def main():
    initial_model = Generator(
                256, 512, 8, channel_multiplier=2,
            ).to(device)




    ckpt_gen="/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/pretrained_checkpoint/200000.pt"

    ckpt = torch.load(ckpt_gen)

    initial_model.load_state_dict(ckpt["g_ema"],strict=False)
    final_model = Generator(
                    256, 512, 8, channel_multiplier=2,
                ).to(device)




    ckpt_gen="/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_AFFHQ/wandb/offline-run-20240420_210442-2e26gxfv/files/No_cat_10001.001500.pt"

    ckpt = torch.load(ckpt_gen)

    final_model.load_state_dict(ckpt["g_ema"])

    eval(final_model,"cat")
    # wandb.init(project="Unlearning_Stylegan2",name="Evaluating GAN")


if __name__ == "__main__":
    main()