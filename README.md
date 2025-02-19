# 🧠 Adapt then Unlearn: Exploring Parameter Space Semantics for Unlearning in Generative Adversarial Networkss  

**Enabling Selective Forgetting in Generative Models**  
*[Arxiv](https://arxiv.org/abs/2309.14054)* | *[TMLR](https://openreview.net/forum?id=jAHEBivObO)* | ![License](https://img.shields.io/badge/License-MIT-blue.svg)

<!-- ![Teaser Image](https://via.placeholder.com/800x300.png?text=GAN+Unlearning+Visual+Demo)  
*Example of feature unlearning in CelebA-HQ (Smiling → Neutral)* -->

---

## 📖 Table of Contents
- [Introduction](#-introduction)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Prerequisites](#-prerequisites)
- [Quick Start and Usage](#-quick-start)
- [Unlearned Models Checkpoint](#-pre-trained-models)
- [Citation](#-citation)

---

## 🌟 Introduction
This repository implements **"Adapt then Unlearn"** – a  framework for selective forgetting in Generative Adversarial Networks (GANs). By exploiting parameter space semantics, our method enables:
- **Class-level unlearning** (e.g., remove digit "7" from MNIST generators)
- **Feature-level unlearning** (e.g., suppress "Smiling" attribute in CelebA-HQ)

*Recommended for researchers exploring machine unlearning, GAN safety, and model editing.*

---

## 🚀 Key Features
- ✅ **Multi-Dataset Support**: MNIST | CelebA-HQ | AFHQ
- ✅ **Three Repulsion Loss Variants**: EL2 | NL2 | IL2
- ✅ **Pretrained Model Integration**
- ✅ **Extensible Codebase**

---

## 💻 Installation
```bash
git clone https://github.com/yourusername/adapt-then-unlearn.git
cd adapt-then-unlearn
conda create -n unlearn python=3.8
conda activate unlearn
pip install -r requirements.txt
```

## Prerequisites
Download required checkpoints:

```bash
# MNIST StyleGAN2
wget https://example.com/pretrained/mnist_generator.pth

# CelebA-HQ StyleGAN2
wget https://example.com/pretrained/celeba_generator.pth

# Classifiers
./scripts/download_classifiers.sh
```

## ⚡ Quick Start and 🛠 Usage

### 🤖 Classifiers
We utilize pre-trained classifiers for the feedback-based framework.

**MNIST**

The checkpoint and model for the MNIST classifier are available in this repository. We used the checkpoints provided in [csinva/gan-vae-pretrained-pytorch](https://github.com/csinva/gan-vae-pretrained-pytorch/tree/master/mnist_classifier).

**CelebA-HQ**

For training and implementation of a classifier for classifying facial attributes in the CelebA dataset, we referred to [rgkannan676/Recognition-and-Classification-of-Facial-Attributes](https://github.com/rgkannan676/Recognition-and-Classification-of-Facial-Attributes/tree/main).


### Stage 1: Negative Adaptation

**MNIST (Digit Class)**:
```bash
python stylegan2_ewc.py --exp "7" --iter 2000 \
  --g_ckpt ./pretrained/mnist_generator.pth --size 32
```

**CelebA-HQ (Facial Feature)**:
```bash
python stylegan2_ewc_train.py --exp "Smiling" --iter 5000 \
  --gan_ckpt ./pretrained/celeba_generator.pth
```

### Stage 2: Unlearning

**MNIST (Class Removal)**:
```bash
python train_different_losses.py --expt "7" \
  --list_of_models adapted_model1.pth adapted_model2.pth \
  --size 32 --ckpt ./pretrained/mnist_generator.pth \
  --iter 3000 --repel --gamma 1.2 --loss_type "l2"
```

**CelebA-HQ (Feature Removal)**:
```bash
python unlearn_main.py --expt "Smiling" \
  --list_of_models classifier1.pth classifier2.pth \
  --ckpt ./pretrained/celeba_generator.pth \
  --iter 5000 --repel_loss --gamma 0.8 --loss_type "exp"
```

**Parameters:**

*   `expt`: Experiment name.
*   `list_of_models`: Paths to the negative adapted models.
*   `ckpt`: Path to the pre-trained GAN.
*   `iter`: Number of iterations.
*   `repel_loss`: Flag to use repulsion loss.
*   `gamma`: Value of the scaling factor for the loss function.
*   `loss_type`: Type of repulsion loss function ("exp", "l2", or "reci").

**Feature List of CelebA-HQ**

| Attribute            | Index | Attribute            | Index | Attribute         | Index | Attribute           | Index | Attribute          | Index |
| :--------------------- | :---- | :--------------------- | :---- | :------------------ | :---- | :-------------------- | :---- | :------------------- | :---- |
| 5_o_Clock_Shadow     | 0     | Arched_Eyebrows      | 1     | Attractive        | 2     | Bags_Under_Eyes    | 3     | Bald               | 4     |
| Bangs                | 5     | Big_Lips             | 6     | Big_Nose           | 7     | Black_Hair         | 8     | Blond_Hair         | 9     |
| Blurry               | 10    | Brown_Hair           | 11    | Bushy_Eyebrows     | 12    | Chubby             | 13    | Double_Chin        | 14    |
| Eyeglasses           | 15    | Goatee               | 16    | Gray_Hair          | 17    | HeavyMakeup        | 18    | HighCheekbones     | 19    |
| Male                 | 20    | MouthSlightlyOpen    | 21    | Mustache           | 22    | NarrowEyes         | 23    | NoBeard            | 24    |
| OvalFace             | 25    | PaleSkin             | 26    | PointyNose         | 27    | RecedingHairline   | 28    | RosyCheeks         | 29    |
| Sideburns            | 30    | Smiling              | 31    | StraightHair       | 32    | WavyHair           | 33    | WearingEarrings    | 34    |
| WearingHat           | 35    | WearingLipstick      | 36    | WearingNecklace     | 37    | WearingNecktie     | 38    | Young              | 39    |


*Note Regarding the Type of Repulsion Loss Function*:

*   For $\mathcal{L}_{repulsion}^{{EL2}}$, use `args.loss_type="exp"`
*   For $\mathcal{L}_{repulsion}^{{NL2}}$, use `args.loss_type="l2"`
*   For $\mathcal{L}_{repulsion}^{{IL2}}$, use `args.loss_type="reci"`

*Theory details in our paper.*

## 📦 Unlearned Models Checkpoint

Access pre-trained unlearned models via the following link:

[Unlearned Models Folder](https://mega.nz/folder/aJ5QRZ5D#R373C4YIRzFtmd2rxTY2sA)

The folder contains subdirectories for CelebA-HQ, MNIST, and AFHQ, each containing (unlearnt) models that can be used for evaluation and comparison.

## ✍️ Citation

If you use this code in your research, please cite our paper:
```bibtex
@article{
tiwary2025adapt,
title={Adapt then Unlearn: Exploring Parameter Space Semantics for Unlearning in Generative Adversarial Networks},
author={Piyush Tiwary and Atri Guha and Subhodip Panda and Prathosh AP},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=jAHEBivObO},
note={}
}
```
