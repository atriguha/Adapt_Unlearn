import argparse
import pickle
import classifier_models
import torch
from classifier_models import CustomResNet50
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
from torch.utils.data import DataLoader,Dataset,Subset
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
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)




def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)
    
device='cuda' 




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
        if(batch==0):
            continue
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
    parser.add_argument("--feature",type=str)
    parser.add_argument("--folder_path",type=str)
    parser.add_argument("--path", type=str)
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
        default=10000,
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
    
# Create a data loader to iterate over the dataset
    
    
    folder_path = args.folder_path  # Replace with your folder path
    extension = ".pt"
    file_paths = copy_file_paths(folder_path, extension)

    # Print the file paths
    print("files read")
    file_paths.sort()
    for path in file_paths:
        print(path)
    result=[]
    fids=[]
    ckpt_list=file_paths
    print(ckpt_list)
    fids=[]
    # retrain_gen = Generator(args.size, 512, 8).to(device)
    # retrain_ckpt="/home/ece/hdd/Piyush/Stylegan2/stylegan2-pytorch/checkpoints_noeyeglasses/390000.pt"
    # ckpt=torch.load(retrain_ckpt)
    # retrain_gen.load_state_dict(ckpt["g_ema"])
    # retrain_gen = nn.DataParallel(retrain_gen)
    # retrain_gen.eval()

    # if args.truncation < 1:
    #     with torch.no_grad():
    #         mean_latent = retrain_gen.mean_latent(args.truncation_mean)

    # else:
    #     mean_latent = None

    


    # features = extract_feature_from_samples(
    #     retrain_gen, inception, args.truncation, mean_latent, args.batch, args.n_sample, device
    # ).numpy()
    # print(f"extracted {features.shape[0]} features")
    # real_mean = np.mean(features, 0)
    # real_cov = np.cov(features, rowvar=False)
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Resize((256,256)),

            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    args.path='/home/ece/hdd/Piyush/Unlearning-EBM/Unlearning_EWC_Loss/afhq/afhq'
# Load the CIFAR-10 dataset
    train_data_path = F'{args.path}/train'
    # val_data_path = 'dataset/afhq/val'

    # Create datasets
    train_dataset = ImageFolder(train_data_path, transform=transform)
    # val_dataset = datasets.ImageFolder(val_data_path, transform=transform)

    

    # Create data loaders

    # Define a function to filter out samples with label 2
    def filter_dataset(dataset, label_to_remove):
        indices = [i for i in range(len(dataset)) if dataset[i][1] != label_to_remove]
        return Subset(dataset, indices)

    # Filter out samples with label 2
    feature_dict={'cat':0, 'dog':1, 'wild':2}
    

    filtered_dataset = filter_dataset(train_dataset, label_to_remove=feature_dict[args.feature])

    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, index):
            image, _ = self.dataset[index]  # Ignore the label
            return image

        def __len__(self):
            return len(self.dataset)

# Create a new DataLoader with only image tensors
    loader_no_bangs = DataLoader(
        ImageDataset(filtered_dataset),
        batch_size=args.batch,
        shuffle=True,
        drop_last=True
    )
    
    

    features = extract_feature_from_samples_data(
        inception,loader_no_bangs
    ).numpy()
    print(f"extracted {features.shape[0]} features")

    # del dataset_no_bangs,loader_no_bangs

    real_mean = np.mean(features, 0)
    real_cov = np.cov(features, rowvar=False)
    
    for ckpt in ckpt_list:
        g = Generator(args.size, 512, 8).to(device)
        ckpt=torch.load(ckpt)
        ckpt["g_ema"] = {k.replace('module.', ''): v for k, v in ckpt["g_ema"].items()}
        g.load_state_dict(ckpt["g_ema"])
        # g.load_state_dict(ckpt)

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

        print("issue in calc_fid")
        
        
        
        
        fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
        # print(ckpt)
        
        print("fid:", fid)
        fids.append(fid)

save_list_to_text_file(fids,args.folder_path,"fid2.txt")
print(fids)