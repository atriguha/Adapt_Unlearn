import sys

from Classifier_Mnist.lenet import LeNet5
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
# def get_checkpoint():
#     # checkpoint = torch.load(
#     # 			"/home/ece/hdd/Piyush/Unlearning-EBM/model_best40.pt")
#     checkpoint=torch.load("/home/ece/hdd/Piyush/Unlearning-EBM/classifier/checkpoints/model40checkpoint_loss.pt")
#     checkpoint_ = {}

#     for key in checkpoint.keys():
#         new_key = key.replace("module.","")
#         checkpoint_[new_key] = checkpoint[key]
#     return checkpoint_

classifier_checkpoint = "Unlearning_MNIST/Classifier_Mnist/lenet_epoch=12_test_acc=0.991.pth"
def img_classifier(images,noise,feature_type):
    # global classifier_checkpoint
    device = "cuda"
    # temp = []
    images=images[0]
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

    
def get_mean_style(generator, device):
    mean_style = None
    for i in range(10):
        style = generator.mean_style(torch.randn(1024, 512).to(device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style

def gen_images_tot(G,feature_type,tot_samples=5000):
    device= "cuda"
    
    # mean_style = get_mean_style(G, device)
    neg_images = []
    pos_images = []
    loop_len=int(tot_samples/100)
    cnt=0

    # while(loop_len>cnt):
    #     cnt+=1

         
    for i in tqdm(range(loop_len)):
        z_latent=mixing_noise(100, 128, 0.9, 'cuda')
        # z_latent = torch.randn(100, 512).to(device)
        gen_img = G(z_latent)
        # print(gen_img[0].shape)
        neg,pos,neg_noise = img_classifier(gen_img,z_latent,feature_type)
        # print(neg.shape)

        # neg, pos = neg.detach().cpu(), pos.detach().cpu()
        neg_images.extend(neg)
        pos_images.extend(pos)
        # print("Neg size", neg.size())
        # print("Pos size",neg.size()[0])

    if tot_samples%100!=0:


        z_latent = mixing_noise(tot_samples%100, 128, 0.9, 'cuda')
        gen_img = G(z_latent)

        neg,pos,neg_noise = img_classifier(gen_img,z_latent,feature_type)
        neg, pos = neg.detach().cpu(), pos.detach().cpu()
        neg_images.extend(neg)
        pos_images.extend(pos)
    
    return neg_images, pos_images

def gen_images_tot_pos(G,feature_type,tot_samples=5000):
    device= "cuda"
    
    # mean_style = get_mean_style(G, device)
    neg_images = []
    pos_images = []
    loop_len=int(tot_samples/10)
    while(len(pos_images)<tot_samples):
         
        for i in tqdm(range(loop_len)):
            z_latent=mixing_noise(10, 128, 0.9, 'cuda')
            # z_latent = torch.randn(100, 512).to(device)
            gen_img = G(z_latent)

            neg,pos,neg_noise = img_classifier(gen_img,z_latent,feature_type)
            neg, pos = neg.detach().cpu(), pos.detach().cpu()
            neg_images.extend(neg)
            pos_images.extend(pos)
            # print("Neg size", neg.size())
            # print("Pos size",neg.size()[0])

    # if tot_samples%10!=0:


    #     z_latent = torch.randn(tot_samples%10, 512).to(device)
    #     gen_img = G(z_latent)

    #     neg,pos,neg_noise = img_classifier(gen_img,z_latent)
    #     neg, pos = neg.detach().cpu(), pos.detach().cpu()
    #     neg_images.extend(neg)
    #     pos_images.extend(pos)
    
    return neg_images, pos_images

def gen_images_tot_neg(G,feature_type,tot_samples=5000):
    device= "cuda"
    
    # mean_style = get_mean_style(G, device)
    neg_images = []
    pos_images = []
    loop_len=int(tot_samples/10)
    while(len(neg_images)<tot_samples):
         
        for i in tqdm(range(loop_len)):
            z_latent=mixing_noise(10, 128, 0.9, 'cuda')
            # z_latent = torch.randn(100, 512).to(device)
            gen_img = G(z_latent)

            neg,pos,neg_noise = img_classifier(gen_img,z_latent,feature_type)
            # neg, pos = neg.detach().cpu(), pos.detach().cpu()
            neg_images.extend(neg)
            pos_images.extend(pos)
            # print("Neg size", neg.size())
            # print("Pos size",neg.size()[0])

    # if tot_samples%10!=0:


    #     z_latent = torch.randn(tot_samples%10, 512).to(device)
    #     gen_img = G(z_latent)

    #     neg,pos,neg_noise = img_classifier(gen_img,z_latent)
    #     neg, pos = neg.detach().cpu(), pos.detach().cpu()
    #     neg_images.extend(neg)
    #     pos_images.extend(pos)
    
    return neg_images, pos_images


# def gen_images_custom(G,pos_samples=5000, neg_samples=5000):




def gen_images_custom(G,pos_samples=5000, neg_samples=5000, interpolate=True):
    device= "cuda"
    mean_style = get_mean_style(G, device)
    neg_images = []
    pos_images = []
    
    while(len(pos_images)<pos_samples or len(neg_images)<neg_samples):
        z_latent = torch.randn(100, 128).to(device)
        gen_img = G(z_latent, step=6, alpha=1,mean_style=mean_style,style_weight=0.7)

        neg,pos,neg_noise = img_classifier(gen_img,z_latent)
        neg, pos = neg.detach().cpu(), pos.detach().cpu()
        neg_images.extend(neg)
        pos_images.extend(pos)
        # print("Neg size", neg.size())
        # print("Pos size",neg.size()[0])
        if(interpolate):

            req_no = 100 - neg.size()[0]
            # print(f"req_no {req_no}")
            neg_latents = []
            for i in range(req_no):
                weight = i / (req_no - 1)
                ind1 = random.randint(0,len(neg_noise)-1)
                ind2 = random.randint(0, len(neg_noise)-1)				
                interpolated_vector = torch.lerp(neg_noise[ind1], neg_noise[ind2], weight)
                neg_latents.append(interpolated_vector)
            neg_latent = torch.stack(neg_latents,dim=0)
            # print(neg_latent.shape, len(neg_latents))

            neg_gen_imgs = G(neg_latent, step=6, alpha=1,mean_style=mean_style,style_weight=0.7)
            i_neg,_,_ = img_classifier(neg_gen_imgs,neg_latent)
            i_neg = i_neg.detach().cpu()
            # print("i_neg size=", i_neg.size())
            neg_images.extend(i_neg)

        if(len(pos_images)>=pos_samples):
            pos_images = pos_images[:pos_samples]


        # print(len(pos_images),len(neg_images))
        if len(neg_images) >= neg_samples and len(pos_images) >= pos_samples:
            neg_images = neg_images[:neg_samples]
            pos_images = pos_images[:pos_samples]
            break
    print(len(neg_images),len(neg_images))
    return neg_images, pos_images

class FeedbackData(data.Dataset):

    def __init__(self, G,sampling_type=2,pos_samples=5000,neg_samples=5000,tot_samples=5000):
        if(sampling_type==1):

            neg_images, pos_images = gen_images_custom(G,pos_samples,neg_samples)

        if(sampling_type==2):
            neg_images, pos_images = gen_images_tot(G,tot_samples)

        if(sampling_type==3):
            print("sampling type",sampling_type)
            neg_images, pos_images = gen_images_custom(G,pos_samples,neg_samples, interpolate=False)

        # print(len(neg_images), len(pos_images))
        neg_images = torch.stack(neg_images,dim=0)
        pos_images = torch.stack(pos_images, dim=0)
        print(neg_images.shape, pos_images.shape)
        self.neg_img = neg_images
        self.pos_img = pos_images

    def __getitem__(self, index):
        pos = self.pos_img[index]
        # a1, a2 = np.random()
        ind1 = random.randint(0,len(self.neg_img)-1)
        ind2 = random.randint(0, len(self.neg_img)-1)
        neg1 = self.neg_img[ind1]
        neg2=self.neg_img[ind2]


        return neg1

        

    def __len__(self):
        return len(self.pos_img)



class FeedbackData_neg(data.Dataset):

    def __init__(self, G,feature_type,sampling_type=2,pos_samples=5000,neg_samples=5000,tot_samples=5000, ind=0):
        

        
        neg_images, pos_images = gen_images_tot(G,feature_type,tot_samples)

       

        print(len(neg_images), len(pos_images))
        neg_images = torch.stack(neg_images,dim=0)
        pos_images = torch.stack(pos_images, dim=0)
        print(neg_images.shape, pos_images.shape)
        
        utils.save_image(
                        neg_images[0:64],
                        f"%s/{feature_type}/{str(len(neg_images)).zfill(6)}_{str(ind)}.png" % ("train_samples"),
                        nrow=int(len(neg_images[0:64]) ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
        self.neg_img = neg_images
        

    def __getitem__(self, index):
        neg = self.neg_img[index]
        
        


        return neg

        

    def __len__(self):
        return len(self.neg_img)
    


class FeedbackData_pos(data.Dataset):

    def __init__(self, G,feature_type,sampling_type=2,pos_samples=5000,neg_samples=5000,tot_samples=5000, ind=0):
        

        
        neg_images, pos_images = gen_images_tot(G,feature_type,tot_samples)

       

        # print(len(neg_images), len(pos_images))
        neg_images = torch.stack(neg_images,dim=0)
        pos_images = torch.stack(pos_images, dim=0)
        print(neg_images.shape, pos_images.shape)
        
        utils.save_image(
                        pos_images[0:64],
                        f"%s/{feature_type}/{str(len(pos_images)).zfill(6)}_{str(ind)}.png" % ("train_samples"),
                        nrow=int(len(neg_images[0:64]) ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
        self.pos_img = pos_images
        

    def __getitem__(self, index):
        pos = self.pos_img[index]
        
        


        return pos

        

    def __len__(self):
        return len(self.pos_img)