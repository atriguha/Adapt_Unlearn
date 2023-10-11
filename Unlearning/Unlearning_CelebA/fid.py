import argparse
import pickle
import classifier_models
import torch
from torch import nn
import numpy as np
from scipy import linalg
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.transforms import Grayscale
from torchvision import transforms, utils
import os
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset
from model import Generator
from calc_inception import load_patched_inception_v3
import torch
import torch.utils.data as data
import torchvision.models as models
from PIL import Image
import os
import os.path


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CelebA(data.Dataset):
    def __init__(self, root, ann_file, transform=None, target_transform=None, loader=default_loader):
        images = []
        targets = []
        
        for line in open(os.path.join(root, ann_file), 'r'):
            sample = line.split(",")
            # print(sample)
            # print(len(sample))
            # if len(sample) != 41:
            #     raise(RuntimeError("# Annotated face attributes of CelebA dataset should not be different from 40"))
            images.append(sample[0])
            targets.append([int(i) for i in sample[1:]])
        self.images = [os.path.join(root, 'img_align_celeba', img) for img in images]
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
		
    def __getitem__(self, index):
        path = self.images[index]
        sample = self.loader(path)
        target = self.targets[index]
        target = torch.LongTensor(target)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.images)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)
    
device='cuda' 

def get_checkpoint():
    # checkpoint = torch.load(
    # 			"/Unlearning-EBM/model_best40.pt")
    checkpoint=torch.load("/Stylegan2/classifier_celeba/model40checkpoint_accuracy.pt")
    checkpoint_ = {}

    for key in checkpoint.keys():
        new_key = key.replace("module.","")
        checkpoint_[new_key] = checkpoint[key]
    return checkpoint_

classifier_checkpoint = get_checkpoint()
@torch.no_grad()
def img_classifier(images,noise,feature_type):
    global classifier_checkpoint
    device = "cuda"
    # temp = []
    
    classifier = classifier_models.resnet50(False)
    classifier.load_state_dict(classifier_checkpoint)

    classifier = classifier.requires_grad_(False).to(device)

    classifier.requires_grad_(False)
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
        for attr_scores in outputs:
            # print(attr_scores)
            _, attr_preds = attr_scores.topk(maxk, 1, True, True)
            top_k_preds.append(attr_preds.t())
    all_preds = torch.cat(top_k_preds, dim=0)
    #{'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, 'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, 'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}
    feature_dict={'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2, 'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19, 'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34, 'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}
    # print(type(feature_type))
    if(feature_type=="Beard"):
         
        ind=feature_dict["No_Beard"]

    else:
         ind=feature_dict[feature_type]
    # print("feature index is:",ind)
    neg = all_preds[ind] #Index: 20; because we're suppresing Male
    
    if(feature_type=="Beard"):
         
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
    return neg_images, pos_images,neg_noise
# resnext50_32x4d = models.resnext50_32x4d(pretrained=False)
# resnext50_32x4d.fc = nn.Linear(2048, 40)
# resnext50_32x4d.to(device)
# path_toLoad="/Unlearning-EBM/classifier/classifier_github/model/model_3_epoch.pt"
# checkpoint = torch.load(path_toLoad)
# resnext50_32x4d.load_state_dict(checkpoint['model_state_dict'])





#Initializing the model with the model parameters of the checkpoint.
# def img_classifier40(images,feature_type=None):
#     global classifier_checkpoint
#     device = "cuda"
#     # temp = []

#     with torch.no_grad():
          
        
#         #Setting the model to be in evaluation mode. This will set the batch normalization parameters.
#         resnext50_32x4d.eval() 
#         res=transforms.Resize((218,178))
#         images=res(images)
#         # ip=torch.randn(8,3,218,178).to(device)
#         scores=resnext50_32x4d(images)
#         # labels=torch.zeros(8,40).to(device)
#         converted_Score=scores.clone()
#         converted_Score[converted_Score>=0]=1
#         converted_Score[converted_Score<0]=0
#         # print(converted_Score)
#         converted_Score=converted_Score.t()

        
#     neg=converted_Score[5]
#     neg_image = []
#     pos_image = []

    
    
    
    

    
#     neg_index = torch.where(neg == 1)[0]
#     pos_index = torch.where(neg == 0)[0]
#     # neg_index = torch.where(neg == 0)[0]
#     # pos_index = torch.where(neg == 1)[0]
#     # print(type(images),type(neg_index))
#     neg_image.append(images[neg_index])
#     pos_image.append(images[pos_index])

#     neg_images = torch.cat(neg_image, dim=0)
#     pos_images = torch.cat(pos_image, dim=0)
    
#     # print(all_preds)
#     return converted_Score,neg_images, pos_images,neg_index

@torch.no_grad()
def extract_feature_from_samples(
    generator, inception, truncation, truncation_latent, batch_size, n_sample, device
):
    print("yoo, we are here to debug")
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    for batch in tqdm(batch_sizes):
        latent = torch.randn(batch, 512, device=device)
        img, _ = generator([latent], truncation=truncation, truncation_latent=truncation_latent)
        feat = inception(img)[0].view(img.shape[0], -1)
        feat=feat.to('cpu')
        features.append(feat)

    features = torch.cat(features, 0)

    return features



def extract_feature_from_samples_data(
    inception, dataloader
):
    
    features = []

    for img in (dataloader):
        # latent = torch.randn(batch, 512, device=device)
        # img, _ = g([latent], truncation=truncation, truncation_latent=truncation_latent)
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to("cpu"))

    features = torch.cat(features, 0)

    return features

def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print("product of cov matrices is singular")
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f"Imaginary component {m}")

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid

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
    parser.add_argument("--path", type=str)
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
        default=20000,
        help="number of the samples for calculating FID",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for generator"
    )
    
    parser.add_argument(
        "--ckpt", type=str
    )
    # parser.add_argument(
    #     "--inception",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="path to precomputed inception embedding",
    # )
    
    args = parser.parse_args()
    
    args.distributed=0
    
    

    
    inception = nn.DataParallel(load_patched_inception_v3()).to(device)
    inception.eval()
    
    
    import torch.utils.data
    import torchvision.transforms as transforms
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Resize((256,256)),

            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    
    celeba_dataset = ImageFolder(root="/Stylegan2/stylegan2-pytorch/CelebA-HQ", transform=transform)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    # celeba_dataset = CelebA(
    #         "/Unlearning-EBM/classifier/dataset",
    #         'test_attr_list.txt',
    #         transforms.Compose([
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor()
                
    #         ]))
    
    # print(len(celeba_dataset))
    ##filtering based on our experiment
    
    # celeba_dataset = [data for data in celeba_dataset if data[1][5] != 1]
    # print(len(celeba_dataset))
# Create a data loader to iterate over the dataset
    
    
    # folder_path = "/Stylegan2/stylegan2-pytorch/checkpoints_nobangs"  # Replace with your folder path
    # extension = ".pt"
    # file_paths = copy_file_paths(folder_path, extension)

    # # Print the file paths
    # print("files read")
    # file_paths.sort()
    # for path in file_paths:
    #     print(path)
    # result=[]
    # fids=[]
    # ckpt_list=file_paths
    # print(ckpt_list)
    fids=[]
    retrain_gen = Generator(args.size, 512, 8).to(device)
    retrain_ckpt="/Stylegan2/stylegan2-pytorch/checkpoints_noeyeglasses/390000.pt"
    ckpt=torch.load(retrain_ckpt)
    retrain_gen.load_state_dict(ckpt["g_ema"])
    retrain_gen = nn.DataParallel(retrain_gen)
    retrain_gen.eval()

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = retrain_gen.mean_latent(args.truncation_mean)

    else:
        mean_latent = None

    


    features = extract_feature_from_samples(
        retrain_gen, inception, args.truncation, mean_latent, args.batch, args.n_sample, device
    ).numpy()
    print(f"extracted {features.shape[0]} features")
    real_mean = np.mean(features, 0)
    real_cov = np.cov(features, rowvar=False)
    # loader = torch.utils.data.DataLoader(celeba_dataset, batch_size=args.batch, sampler=data_sampler(celeba_dataset, shuffle=True, distributed=args.distributed),drop_last=True)
    # pos_images=[]
    # # print(len(pos_images))
    # for img,_ in (loader):
    #     img=img.to('cuda')
    #     # pos=img
    #     _,pos, _=img_classifier(img,noise=None,feature_type="Bangs")
    #     pos=pos.detach().cpu()
    #     # print("new images",pos.shape)
    #     pos_images.extend(pos)
    #     # print("updated size", len(pos_images))
    
    # grid = utils.make_grid(pos_images[0:64], nrow=8, normalize=True, range=(0,1))
    # utils.save_image(grid,"pos_images_dataset.png",nrow=int(64 ** 0.5),
    #                         normalize=False,
    #                         range=(-1, 1),)

    # class ImageDataset(Dataset):
    #     def __init__(self, image_list, transform=None):
    #         self.image_list = image_list
            

    #     def __len__(self):
    #         return len(self.image_list)

    #     def __getitem__(self, index):
            
    #         image=self.image_list[index]
    #         return image
    # dataset_no_bangs = ImageDataset(pos_images, transform=None)
    # loader_no_bangs = torch.utils.data.DataLoader(dataset_no_bangs, batch_size=args.batch, sampler=data_sampler(dataset_no_bangs, shuffle=True, distributed=args.distributed),drop_last=True)
    # del loader,celeba_dataset,

    # features = extract_feature_from_samples_data(
    #     inception,loader_no_bangs
    # ).numpy()
    # print(f"extracted {features.shape[0]} features")

    # # del dataset_no_bangs,loader_no_bangs

    # real_mean = np.mean(features, 0)
    # real_cov = np.cov(features, rowvar=False)
    # ckpt_list=["/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.0.pt",
    #            "/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.1.pt",
    #            "/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.2.pt",
    #            "/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-0.30000000000000004.pt",
    #            "/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.4.pt",
    #            "/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.5.pt",
    #            "/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-0.6000000000000001.pt",
    #            "/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-0.7000000000000001.pt",
    #            "/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.8.pt",
    #            "/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.9.pt",
    #            "/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-001.0.pt",
    #            "/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-001.1.pt",
    #            "/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-1.2000000000000002.pt",
    #            "/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-001.3.pt"
               

               
               
    #            ]
    ##provide path to models to  be evaluated
    ckpt_list=["/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.5.pt",
               "/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.5.pt",
               "/Stylegan2/unlearning_gan/Unlearning_CelebA/checkpoints/Eyeglasses/Unlearning/-000.5.pt"]
    for ckpt in ckpt_list:
        g = Generator(args.size, 512, 8).to(device)
        ckpt=torch.load(ckpt)
        g.load_state_dict(ckpt)
        g = nn.DataParallel(g)
        g.eval()

        if args.truncation < 1:
            with torch.no_grad():
                mean_latent = g.mean_latent(args.truncation_mean)

        else:
            mean_latent = None

        


        features = extract_feature_from_samples(
            g, inception, args.truncation, mean_latent, args.batch, args.n_sample, device
        ).numpy()
        print(f"extracted {features.shape[0]} features")
        sample_mean = np.mean(features, 0)
        sample_cov = np.cov(features, rowvar=False)

        
        
        
        
        fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
        # print(ckpt)
        
        print("fid:", fid)
        fids.append(fid)


print(fids)