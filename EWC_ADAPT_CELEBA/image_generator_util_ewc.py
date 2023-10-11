import sys

import classifier_models
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
#     # 			"/Unlearning-EBM/model_best40.pt")
#     checkpoint=torch.load("/Unlearning-EBM/classifier/checkpoints/model40checkpoint_loss.pt")
#     checkpoint_ = {}

#     for key in checkpoint.keys():
#         new_key = key.replace("module.","")
#         checkpoint_[new_key] = checkpoint[key]
#     return checkpoint_

# classifier_checkpoint = get_checkpoint()
# def img_classifier(images,noise,feature_type):
#     global classifier_checkpoint
#     device = "cuda"
#     # temp = []
#     images=images[0]
#     classifier = classifier_models.resnet50(False)
#     classifier.load_state_dict(classifier_checkpoint)

#     classifier = classifier.requires_grad_(False).to(device)

#     classifier.requires_grad_(False)
#     maxk = 1
#     neg_image = []
#     pos_image = []
#     neg_noise = []
#     # data = TensorDataset(images,noise)
    
#     # print(data_loader.__dict__)
#     # for data in data_loader:
#     # 	data = data[0]
#     top_k_preds = []
#     with torch.no_grad():

#         outputs = classifier(images)  
#         for attr_scores in outputs:
#             # print(attr_scores)
#             _, attr_preds = attr_scores.topk(maxk, 1, True, True)
#             top_k_preds.append(attr_preds.t())
#     all_preds = torch.cat(top_k_preds, dim=0)
#     #{'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, 'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, 'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}
#     feature_dict={'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, 'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, 'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}
#     # print(type(feature_type))
#     if(feature_type=="Beard"):
         
#         ind=feature_dict["No_Beard"]

#     else:
#          ind=feature_dict[feature_type]
#     # print("feature index is:",ind)
#     neg = all_preds[ind] #Index: 20; because we're suppresing Male
    
#     if(feature_type=="Beard"):
         
#         pos_index = torch.where(neg ==1)[0]
#         neg_index = torch.where(neg == 0)[0]

#     else:
#          neg_index = torch.where(neg ==1)[0]
#          pos_index = torch.where(neg == 0)[0]
#     neg_image.append(images[neg_index])
#     # neg_noise.append(noise[neg_index])
#     # pos_noise.append(noise[pos])
#     pos_image.append(images[pos_index])
#     # neg_noise = torch.cat(neg_noise, dim=0)
#     neg_images = torch.cat(neg_image, dim=0)
#     pos_images = torch.cat(pos_image, dim=0)
#     return neg_images, pos_images,neg_noise
# device='cuda'

# # classifier_checkpoint = get_checkpoint()
def img_classifier(images,noise,feature_type):
    device = "cuda"
    resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
    resnext50_32x4d.fc = nn.Linear(2048, 40)
    resnext50_32x4d.to(device)
    ##provide classifier path
    path_toLoad="/classifier_celeba/model/model_3_epoch.pt"
    checkpoint = torch.load(path_toLoad)
    
    resnext50_32x4d.load_state_dict(checkpoint['model_state_dict'])

    # global classifier_checkpoint
    
    # temp = []
    
    
    maxk = 1
    neg_image = []
    pos_image = []
    neg_noise = []
    noise=noise[0]
    # print(type(noise))
    # data = TensorDataset(images,noise)
    
    # print(data_loader.__dict__)
    # for data in data_loader:
    # 	data = data[0]
    # print(type(images))
    images=images[0]
    with torch.no_grad():
          
        transform = transforms.Resize((218,178))
        
        images_classifier=transform(images)
        # print(images.shape)
        #Setting the model to be in evaluation mode. This will set the batch normalization parameters.
        resnext50_32x4d.eval() 
        # ip=torch.randn(8,3,218,178).to(device)
        scores=resnext50_32x4d(images_classifier)
        # labels=torch.zeros(8,40).to(device)
        converted_Score=scores.clone()
        converted_Score[converted_Score>=0]=1
        converted_Score[converted_Score<0]=0
        # print(converted_Score)
        converted_Score=converted_Score.t()
    all_preds = converted_Score
    feature_dict={'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, 'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, 'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}
    # print(type(feature_type))
    ind=feature_dict[feature_type]
    # print("feature index is:",ind)

    neg = all_preds[ind] #Index: 20; because we're suppresing Male
    if(feature_type=="No_Beard"):
        # print("yoooooooooooooooooooo")
        pos_index = torch.where(neg ==1)[0]
        neg_index = torch.where(neg == 0)[0]

    else:
         neg_index = torch.where(neg ==1)[0]
         pos_index = torch.where(neg == 0)[0]
    neg_image.append(images[neg_index])
    # neg_noise.append(noise[neg_index])
    # pos_noise.append(noise[pos])
    pos_image.append(images[pos_index])
    # neg_noise = torch.cat(neg_noise, dim=0)
    neg_images = torch.cat(neg_image, dim=0)
    pos_images = torch.cat(pos_image, dim=0)
    if(len(neg_images)+len(pos_images)!=len(images)):
        print("fishyyyyyyyyyyyyyyy")
    return neg_images, pos_images,neg_noise
    


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
    loop_len=int(tot_samples/10)
    cnt=0

    # while(loop_len>cnt):
    #     cnt+=1

         
    for i in tqdm(range(loop_len)):
        z_latent=mixing_noise(64, 512, 0.9, 'cuda')
        # z_latent = torch.randn(100, 512).to(device)
        gen_img = G(z_latent)

        neg,pos,neg_noise = img_classifier(gen_img,z_latent,feature_type)
        neg, pos = neg.detach().cpu(), pos.detach().cpu()
        neg_images.extend(neg)
        pos_images.extend(pos)
        # print("Neg size", neg.size())
        # print("Pos size",neg.size()[0])

    if tot_samples%10!=0:


        z_latent = mixing_noise(tot_samples%10, 512, 0.9, 'cuda')
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
            z_latent=mixing_noise(10, 512, 0.9, 'cuda')
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
    # pos_images = []
    loop_len=500
    while(len(neg_images)<tot_samples):
        print(len(neg_images))
        for i in tqdm(range(loop_len)):
            z_latent=mixing_noise(32, 512, 0.9, 'cuda')
            # z_latent = torch.randn(100, 512).to(device)
            gen_img = G(z_latent)

            neg,_,neg_noise = img_classifier(gen_img,z_latent,feature_type)
            neg = neg.detach().cpu()
            neg_images.extend(neg)
            # pos_images.extend(pos)
            # print("Neg size", neg.size())
            # print("Pos size",neg.size()[0])

    neg_images=neg_images[0:tot_samples]
    
    # if tot_samples%10!=0:


    #     z_latent = torch.randn(tot_samples%10, 512).to(device)
    #     gen_img = G(z_latent)

    #     neg,pos,neg_noise = img_classifier(gen_img,z_latent)
    #     neg, pos = neg.detach().cpu(), pos.detach().cpu()
    #     neg_images.extend(neg)
    #     pos_images.extend(pos)
    
    return neg_images, None


# def gen_images_custom(G,pos_samples=5000, neg_samples=5000):




def gen_images_custom(G,pos_samples=5000, neg_samples=5000, interpolate=True):
    device= "cuda"
    mean_style = get_mean_style(G, device)
    neg_images = []
    pos_images = []
    
    while(len(pos_images)<pos_samples or len(neg_images)<neg_samples):
        z_latent = torch.randn(100, 512).to(device)
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
        

        
        neg_images, _ = gen_images_tot_neg(G,feature_type,tot_samples)

       

        # print(len(neg_images), len(pos_images))
        neg_images = torch.stack(neg_images,dim=0)
        # pos_images = torch.stack(pos_images, dim=0)
        print(neg_images.shape)
        
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