import os
import sys


from Classifier_Mnist.lenet import LeNet5
from model import Generator,Discriminator
import numpy as np
# from torchvision.models import ResNet50_Weights
# from torchvision.models import ResNet50_Weights
import torch
import sys
import torch.backends.cudnn as cudnn
import random
import torch.utils.data as data
import torchvision.models as models
import torch.nn as nn

from torchvision import transforms, utils
# from ebm import EBM

from tqdm import tqdm
@torch.no_grad()

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
        
device='cuda'
def eval(args,final_model,feature_type):
    # wandb.init(project="Unlearning_Stylegan2",name="Evaluating GAN")
    initial_model = Generator(
                args.size, args.latent, 8, channel_multiplier=2,
            ).to(device)




    ckpt_gen=args.ckpt

    ckpt = torch.load(ckpt_gen)

    initial_model.load_state_dict(ckpt["g_ema"],strict=False)
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
    ##latent space for evaluation
    latent_space=torch.load("Unlearning_MNIST/latent_vector_batch64.pt")
    # for i in range(78):
    #     latent_space.append(torch.randn(64,512).to(device))
    
    # latent_space=[sample_z]
    for ind in tqdm(range(len(latent_space))):
        latent_vector=latent_space[ind].to(device)
        gen_img = initial_model([latent_vector])
        op=gen_img[0]
        # print("output",op.shape)
        # norm_range(op,(-1,1))
        neg, pos,negind = img_classifier(op,feature_type)
        
        del gen_img
        del op
        # neg, pos = neg.detach().cpu(), pos.detach().cpu()
        features_gan=[]
        

        

        
        neg_noise.append(latent_vector[negind])
        
        all_negs_old.extend(neg)
        all_pos_old.extend(pos)


        


        if(len(negind)>0):
            gen_img = final_model([latent_vector[negind]])
            op=gen_img[0]
            # norm_range(op,(-1,1))
            
            neg, pos,_ = img_classifier(op,feature_type)
            op=op.detach().cpu()
            all_img_new.extend(op)
            del gen_img
            del op
            # neg, pos = neg.detach().cpu(), pos.detach().cpu()
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




    ckpt_gen="checkpoint/actual_checkpoint.pt"

    ckpt = torch.load(ckpt_gen)

    initial_model.load_state_dict(ckpt["g_ema"],strict=False)
    final_model = Generator(
                    256, 512, 8, channel_multiplier=2,
                ).to(device)




    ckpt_gen="stylegan2/wandb/run-20230620_195635-d7fozq16/files/03000.pt"

    ckpt = torch.load(ckpt_gen)

    final_model.load_state_dict(ckpt["g_ema"],strict=False)

    eval(initial_model,final_model)
    # wandb.init(project="Unlearning_Stylegan2",name="Evaluating GAN")


if __name__ == "__main__":
    main()