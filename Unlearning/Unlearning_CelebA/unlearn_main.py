import argparse
import math
import random
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from submit import _create_run_dir_local, _copy_dir
from image_generator_util_ewc import FeedbackData, FeedbackData_neg,FeedbackData_pos
from torch.utils.data import DataLoader
from eval_newgrad import eval
try:
    import wandb

except ImportError:
    wandb = None


from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment

def calc_scaling_fact(num):
    fact=0
    if(num>1):
        while(num>0):
            num=num//10
            fact+=1

    return fact
def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

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
def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


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


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None
device='cuda'

##provide path to adapted models
ckpt_gens_repel=["/EWC_ADAPT_CELEBA/wandb/run-20230805_172211-f9x0o3ai/files/EWCcelebaHQEyeglasses_0_09000.pt",
               "/EWC_ADAPT_CELEBA/wandb/run-20230805_172218-6q7tt494/files/EWCcelebaHQEyeglasses_0_09000.pt"
               , "/EWC_ADAPT_CELEBA/wandb/run-20230805_224935-hjk84mvr/files/EWCcelebaHQEyeglasses_1_09000.pt"
               ,"/EWC_ADAPT_CELEBA/wandb/run-20230805_224938-scb0fzlx/files/EWCcelebaHQEyeglasses_1_09000.pt",
               "/EWC_ADAPT_CELEBA/checkpoints_5000/Eyeglasses/EWCcelebaHQEyeglasses_0_09000.pt",
              
               
               
               
               ]



def calc_difference_param_new(args,initial_model):
    from model import Generator, Discriminator
    mse=nn.MSELoss()

    
    
    
    ckpt_gens=args.list_of_models





    

    parameter_difference = {}
    for ckpt_gen in ckpt_gens:
        # print(ckpt_gen)
        fine_tuned_model = Generator(
                256, 512, 8, channel_multiplier=2,ckpt_disc=None
            ).to(device)
          


       

        ckpt = torch.load(ckpt_gen)

        fine_tuned_model.load_state_dict(ckpt["g_ema"],strict=False)

        # Assume both models have the same number of parameters and identical shapes
        num_params = sum(p.numel() for p in initial_model.parameters())
        assert num_params == sum(p.numel() for p in fine_tuned_model.parameters())

        # Subtract the parameters of the fine-tuned model from the initial model
        
        cnt=0

        
        for initial_param ,fine_tuned_param in zip(initial_model.named_parameters(), fine_tuned_model.named_parameters()):
            initial_param_name,initial_weight=initial_param
            fine_tuned_name,fine_tuned_weight=fine_tuned_param
            if(initial_param_name==fine_tuned_name):
                cnt+=1
                
                op = mse(fine_tuned_weight,initial_weight)
            # a=torch.square(difference)
            
                try:
                    parameter_difference[initial_param_name]+=op
                except KeyError:
                    parameter_difference[initial_param_name]=op

    val=0
    for itr in parameter_difference:
          
          value=parameter_difference[itr]
          value=value/len(parameter_difference)

          avg_value=value/2*len(ckpt_gens)
          parameter_difference[itr]=avg_value
          val+=avg_value

    del parameter_difference
    # print(parameter_difference)
    return val

def calc_difference_param_new_attract(args,initial_model,ckpt_gens):
    from model import Generator, Discriminator
    mse=nn.MSELoss()

    
    
    





    
    

    parameter_difference = {}
    for ckpt_gen in ckpt_gens:
        # print(ckpt_gen)
        fine_tuned_model = Generator(
                256, 512, 8, channel_multiplier=2,ckpt_disc=None
            ).to(device)
          


       

        ckpt = torch.load(ckpt_gen)

        fine_tuned_model.load_state_dict(ckpt,strict=False)

        # Assume both models have the same number of parameters and identical shapes
        num_params = sum(p.numel() for p in initial_model.parameters())
        
        assert num_params == sum(p.numel() for p in fine_tuned_model.parameters())

        # Subtract the parameters of the fine-tuned model from the initial model
        
        cnt=0

        
        for initial_param ,fine_tuned_param in zip(initial_model.named_parameters(), fine_tuned_model.named_parameters()):
            initial_param_name,initial_weight=initial_param
            fine_tuned_name,fine_tuned_weight=fine_tuned_param
            if(initial_param_name==fine_tuned_name):
                cnt+=1
                
                op = mse(fine_tuned_weight,initial_weight)
            # a=torch.square(difference)
            
                try:
                    parameter_difference[initial_param_name]+=op
                except KeyError:
                    parameter_difference[initial_param_name]=op

    val=0
    for itr in parameter_difference:
          
          value=parameter_difference[itr]
          value=value/len(parameter_difference)

          avg_value=value/2*len(ckpt_gens)
          parameter_difference[itr]=avg_value
          val+=avg_value

    del parameter_difference
    # print(parameter_difference)
    return val



def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)
    param_dict={}
    if(args.ewc_loss):
        param_names,param_dict,output=generator.estimate_fisher(discriminator,param_dict,sample_size=32)
        generator.consolidate(output,param_names,param_dict)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment: 
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred = discriminator(fake_img)
        val=0
        # attract_val=calc_difference_param_new_attract(generator,ckpt_gens_attract)

        val_factor=1
        # if(i<100):
        #     val_factor=100


        
        # else:
        #     val_factor=1


        
        
        

        
        

        

            
        if(args.repel_loss):
            val=calc_difference_param_new(generator,ckpt_gens_repel)

            if(args.scale_repel_loss):

                val_factor=10**calc_scaling_fact(val)
            
            else:
                val_factor=(1/args.gamma)
        ewc_loss=0
        if(args.ewc_loss):
            _,ewc_loss = generator.ewc_loss(param_names,param_dict,cuda=device)

        


            

        g_loss = g_nonsaturating_loss(fake_pred)
        
        # loss=g_loss+250*torch.exp(-(val/val_factor))
        if(args.loss_type=="reci"):

            loss=g_loss+(1/(val*val_factor))
        elif(args.loss_type=="exp"):
            loss=g_loss+100*torch.exp(-(val/val_factor))

        elif(args.loss_type=="l2"):
            loss=g_loss-(val/val_factor)
        
        # loss=g_loss+(1/(val*val_factor))
        # g_loss=g_nonsaturating_loss(fake_pred)
        # if idx%10==0:
        #     val=calc_difference_param_new(generator)
        #     g_loss-=val*args.gamma

        loss_dict["g"] = loss

        generator.zero_grad()
        loss.backward()
        g_optim.step()
        
        # for names,param in generator.named_parameters():
        #     print(param_difference[names].shape)
        #     param.data-=torch.autograd.grad(param_difference[names],param)*args.gamma

        # del param_difference
        
        
        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )
            
            # if wandb and args.wandb:
            wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                        "repel_loss":val,
                        "adversarial loss":g_loss,
                        "ewc_loss":ewc_loss,
                        # "attract_loss":attract_val
                    }
                )

            if i % 100 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z])
                    sample=sample.detach().to('cpu')
                    grid = utils.make_grid(sample, nrow=8, normalize=True, range=(0,1))
                    wandb.log({f'Generated Images_Without {args.exp}': wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=i)
                    
                    

            if i % 500 == 0:
                all_pos_old,all_pos_new,all_negs_new,all_negs_old,all_img_new=eval(generator,args.exp)
                print(f"Images generated by old GAN_Without {args.exp}:","Pos Images:",len(all_pos_old), "Neg Images:", len(all_negs_old))
                print(f"Images generated by new GAN_Without {args.exp}:","Pos Images:",len(all_pos_new), "Neg Images:", len(all_negs_new))
                
                
                if(len(all_negs_old)!=0):
                    grid = utils.make_grid(all_negs_old[0:64], nrow=8, normalize=True, range=(0,1))
                    wandb.log({'Images_old_GAN': wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=i)
         


                if(len(all_img_new)!=0):
                    grid = utils.make_grid(all_img_new[0:64], nrow=8, normalize=True, range=(0,1))
                    wandb.log({'Images_New_GAN': wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=i)
                        # torch.save(model.state_dict(), os.path.join(wandb.run.dir, f'{str(i).zfill(5)}.pt'))
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    os.path.join(wandb.run.dir, f'No_{args.exp}_{args.total_samples}{args.gamma}{str(i).zfill(5)}.pt'),
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")
    parser.add_argument("--gamma",type=float,default=-0.1)
    parser.add_argument("--exp",type=str,default="Bangs")
    # parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    
    parser.add_argument(
        "--iter", type=int, default=40, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument('--list_of_models', nargs='+', help='List of models adapted on EWC loss', required=True)

    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        
        help="mention the type of loss function (reci,l2 or exp)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument(
        "--repel_loss", action="store_true", help="Usage of repel loss in loss function"
    )
    parser.add_argument(
        "--scale_repel_loss", action="store_true", help="Scale Repell_loss between 0 and 10"
    )
    parser.add_argument(
        "--ewc_loss", action="store_true", help="Usage of ewc loss in loss function"
    )
    

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    if args.arch == 'stylegan2':
        from model import Generator, Discriminator

    elif args.arch == 'swagan':
        from swagan import Generator, Discriminator

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt)

        

        generator.load_state_dict(ckpt["g"],strict=False)
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"],strict=False)

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    
    args.total_samples=5000
    
    
    

        

        

    wandb.init(project=f"Unlearning_stylegan 2_CELEBAHQ{args.exp}_{args.total_samples}")
    
    feature_type=args.exp
    args.run_dir = wandb.run.dir
    _copy_dir(['unlearn_main.py','run_file.sh'], args.run_dir)
    dataset = FeedbackData_pos(generator,feature_type,sampling_type=2,tot_samples=args.total_samples,ind=10)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=1, drop_last=True)

    train(args, dataloader, generator, discriminator, g_optim, d_optim, g_ema, device)
    
