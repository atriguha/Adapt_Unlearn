import os
import sys
sys.path.append("/home/ece/hdd/Piyush/Unlearning-EBM")
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from torchvision import transforms, utils
from classifier_models import CustomResNet50,CustomResnet18,CNN
import torchvision.models as models

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
def save_list_to_text_file(my_list, directory, file_name):
    # Ensure the directory exists
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create the full path for the text file
    file_path = os.path.join(directory, file_name)

    # Open the file in write mode and write the list elements
    with open(file_path, 'w') as file:
        for item in my_list:
            file.write(str(item) + '\n')


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
def get_checkpoint():
    # checkpoint = torch.load(
    # 			"/home/ece/hdd/Piyush/Unlearning-EBM/model_best40.pt")
    checkpoint=torch.load("/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/classifier_AFFHQ/CNN/model.pt",map_location='cuda')
    checkpoint_ = {}

    # for key in checkpoint.keys():
    #     new_key = key.replace("module.","")
    #     checkpoint_[new_key] = checkpoint[key]
    return checkpoint

classifier_checkpoint = get_checkpoint()
def img_classifier40(images,feature_type,images_org):
    global classifier_checkpoint
    device = "cuda"
    # temp = []
    images=images
    classifier = CNN()
    # classifier.fc = nn.Linear(2048, 40)
    # classifier =CustomResNet50(in_channels=3,num_classes=3)
    # classifier.load_state_dict(classifier_checkpoint["model_state_dict"])
    classifier.load_state_dict(classifier_checkpoint)


    classifier = classifier.requires_grad_(False).to(device)

    # classifier.requires_grad_(False)
    maxk = 1
    neg_image = []
    pos_image = []
    neg_noise = []
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
        

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Calculate FID scores")
    parser.add_argument("--feature",type=str,required=True)
    parser.add_argument("--folder_path",type=str,required=True)
    parser.add_argument("--truncation", type=float, default=1, help="truncation factor")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of samples to calculate mean for truncation",
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch size for the generator"
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




    ckpt_gen="/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/EWC_ADAPT_AFFHQ/pretrained_checkpoint/200000.pt"

    ckpt = torch.load(ckpt_gen)

    initial_model.load_state_dict(ckpt["g_ema"])
    final_model = Generator(
                    256, 512, 8, channel_multiplier=2,
                ).to(device)



    result=[]
    fids=[]
    
    # ckpt_gens=["/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.0.pt",
    #            "/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.1.pt",
    #            "/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.2.pt",
    #            "/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-0.30000000000000004.pt",
    #            "/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.4.pt",
    #            "/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.5.pt",
    #            "/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-0.6000000000000001.pt",
    #            "/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-0.7000000000000001.pt",
    #            "/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.8.pt",
    #            "/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.9.pt",
    #            "/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-001.0.pt",
    #            "/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-001.1.pt",
    #            "/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-1.2000000000000002.pt",
    #            "/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-001.3.pt"
               

               
               
    #            ]

    # ckpt_gens=["/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_CIFAR10/wandb/run-20240403_014420-29539tsz/files/No_bird_50001.000500.pt"]

    # folder_path = "/home/ece/hdd/Piyush/Stylegan2/unlearning_gan/Unlearning_AFFHQ/wandb/offline-run-20240420_142456-2jjjzlkj"  # Replace with your folder path
    folder_path=args.folder_path
    extension = ".pt"
    file_paths = copy_file_paths(folder_path, extension)

    # Print the file paths
    file_paths.sort()
    # file_paths.reverse()
    # for path in file_paths:
    #     print(path)
    result=[]
    # fids=[]
    ckpt_gens=file_paths
    print(ckpt_gens)
    # ckpt_gens.reverse()
    cnt=0
    counter=0
    for ckpt_gen in ckpt_gens:
        # if(cnt%500!=0):
        #     cnt+=50
            
        #     continue

        # cnt+=50

        ckpt = torch.load(ckpt_gen)
        ckpt["g_ema"] = {k.replace('module.', ''): v for k, v in ckpt["g_ema"].items()}

        final_model.load_state_dict(ckpt["g_ema"])
        # final_model.load_state_dict(ckpt)

        # sample_z=torch.load("/home/ece/hdd/Piyush/Unlearning-EBM/VQ-VAE/stylegan2/sample_z.pt")
            # for names,param in initial_model.named_parameters():
            #     param.data+=param_difference[names]*gamma
        latent_space=[]
        all_negs_old=[]
        all_negs_new=[]
        all_pos_old=[]
        all_pos_new=[]

        all_features_og=[0]*3
        all_features_new=[0]*3
        latent_space=[]
        latent_space=torch.load("latent_vector_Batch64.pt")
        # for i in range(78):
        #     latent_space.append(torch.randn(64,512).to(device))
        
        # latent_space=[sample_z]
        for ind in tqdm(range(len(latent_space))):
            gen_img = initial_model([latent_space[ind]])
            op=gen_img[0]
            norm_range(op,(-1,1))
            resize_transform=transforms.Resize((250,250))
            gen_img_resized=resize_transform(op)
            allnegs,neg, pos,negind = img_classifier40(gen_img_resized,args.feature,op)
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
            gen_img_resized=resize_transform(op)

            allnegs,neg, pos,_ = img_classifier40(gen_img_resized,args.feature,op)
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
                            f"eval_images/{str(ind).zfill(6)}.png",
                            nrow=int(64 ** 0.5),
                            normalize=False,
                            range=(-1, 1),
                        )
        print("Images generated by old GANs:","Pos Images:",len(all_pos_old), "Neg Images:", len(all_negs_old))
        print("Images generated by new GANs:","Pos Images:",len(all_pos_new), "Neg Images:", len(all_negs_new))

        result.append((len(all_negs_old)-len(all_negs_new))*100/len(all_negs_old))
        # grid_images = utils.make_grid(all_negs_new[:64], nrow=8, normalize=True, range=(-1, 1))

        # Specify the path where you want to save the grid image
        # save_path = F'{args.folder_path}/grid_image_{counter}.png'
        counter+=1

        # Save the grid image
        # utils.save_image(grid_images, save_path)
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
    save_list_to_text_file(result,folder_path,"result2.txt")
    
    # df.to_csv(os.path.join(wandb.run.dir, "results_gan_Wearing_Hat_gamma_0.001_total_fixed.csv"))


