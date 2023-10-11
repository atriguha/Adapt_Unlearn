import argparse
import pickle

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
from torch.utils.data import DataLoader
from model import Generator
from calc_inception import load_patched_inception_v3

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)
    
@torch.no_grad()
def extract_feature_from_samples(
    generator, inception, truncation, truncation_latent, batch_size, n_sample, device
):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    for batch in tqdm(batch_sizes):
        latent = torch.randn(batch, 128, device=device)
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

    for img,_ in (dataloader):
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


def fid_func(ckpt,feature_type):
    device='cuda'
    inception = nn.DataParallel(load_patched_inception_v3()).to(device)
    inception.eval()
    transform = transforms.Compose(
        [   Grayscale(num_output_channels=3),
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),


            
        ]
    )
    ##path to MNIST dataset
    path="MNIST/raw/MNIST"
    mnist_dataset = datasets.MNIST(path, train=False, download=True, transform=transform)
    print(len(mnist_dataset))
    ##filtering based on our experiment
    # for data in mnist_dataset:
    #     print(data[1])
    mnist_dataset = [data for data in mnist_dataset if data[1] != feature_type]
    print(len(mnist_dataset))
# Create a data loader to iterate over the dataset
    loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=64, sampler=data_sampler(mnist_dataset, shuffle=True, distributed=0),drop_last=True)
    features = extract_feature_from_samples_data(
        inception,loader
    ).numpy()
    print(f"extracted {features.shape[0]} features")

    del loader,mnist_dataset

    real_mean = np.mean(features, 0)
    real_cov = np.cov(features, rowvar=False)
    g = Generator(32, 128, 8).to(device)
    ckpt=torch.load(ckpt)
    g.load_state_dict(ckpt["g_ema"])
    g = nn.DataParallel(g)
    g.eval()
    truncation=1
    truncation_mean=4096
    if truncation < 1:
        with torch.no_grad():
            mean_latent = g.mean_latent(truncation_mean)

    else:
        mean_latent = None

    

    features = extract_feature_from_samples(
        g, inception, 1, mean_latent, 64, 10000, device
    ).numpy()
    print(f"extracted {features.shape[0]} features")
    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    
    
    
    
    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
    # print(args.ckpt)
    
    return fid





def fid_func2(ckpt1,ckpt2):
    device='cuda'
    inception = nn.DataParallel(load_patched_inception_v3()).to(device)
    inception.eval()

    retrained_gen = Generator(32, 128, 8).to(device)
    ckpt1=torch.load(ckpt1)
    retrained_gen.load_state_dict(ckpt1["g_ema"])
    retrained_gen = nn.DataParallel(retrained_gen)
    retrained_gen.eval()
    truncation=1
    truncation_mean=4096
    if truncation < 1:
        with torch.no_grad():
            mean_latent = g.mean_latent(truncation_mean)

    else:
        mean_latent = None

    

    features = extract_feature_from_samples(
        retrained_gen, inception, 1, mean_latent, 64, 10000, device
    ).numpy()
    print(f"extracted {features.shape[0]} features")
    real_mean = np.mean(features, 0)
    real_cov = np.cov(features, rowvar=False)
    g = Generator(32, 128, 8).to(device)
    ckpt2=torch.load(ckpt2)
    g.load_state_dict(ckpt2["g_ema"])
    g = nn.DataParallel(g)
    g.eval()
    truncation=1
    truncation_mean=4096
    if truncation < 1:
        with torch.no_grad():
            mean_latent = g.mean_latent(truncation_mean)

    else:
        mean_latent = None

    

    features = extract_feature_from_samples(
        g, inception, 1, mean_latent, 64, 10000, device
    ).numpy()
    print(f"extracted {features.shape[0]} features")
    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    
    
    
    
    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
    # print(args.ckpt)
    
    return fid





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
    transform = transforms.Compose(
        [   Grayscale(num_output_channels=3),
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),


            
        ]
    )
    
    
    mnist_dataset = datasets.MNIST(args.path, train=False, download=True, transform=transform)
    print(len(mnist_dataset))
    ##filtering based on our experiment
    
    mnist_dataset = [data for data in mnist_dataset if data[1] != 8]
    print(len(mnist_dataset))
# Create a data loader to iterate over the dataset
    loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=args.batch, sampler=data_sampler(mnist_dataset, shuffle=True, distributed=args.distributed),drop_last=True)
    features = extract_feature_from_samples_data(
        inception,loader
    ).numpy()
    print(f"extracted {features.shape[0]} features")

    del loader,mnist_dataset

    real_mean = np.mean(features, 0)
    real_cov = np.cov(features, rowvar=False)
    g = Generator(args.size, 128, 8).to(device)
    ckpt=torch.load(args.ckpt)
    g.load_state_dict(ckpt["g_ema"])
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
    # print(args.ckpt)
    
    print("fid:", fid)



