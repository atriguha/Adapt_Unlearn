
# Adapt_Unlearn

# Project Name

Brief description or introduction of your project.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  
- [Adaptation](#adaptation)
    - [MNIST_Adaptation ](#adapt_mnist)
    - [CELEBAHQ_Adaptation ](#adapt_celebahq)

- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

Explain what someone needs to do to get started with your project.

### Prerequisites
Required checkpoints for 
1. Pre Trained Stylegan2 on CELEBA-HQ
2. Pre Trained Stylegan2 on MNIST
3. Pre Trained Classifiers

### Adaptation

The implementation for Adaptation is done seperately for MNIST and CELEBA-HQ . They can be found in ADAPT_MNIST and ADAPT_CELEBA folders respectively.

### MNIST_Adaptation 
To adapt to a certain class in MNIST
``` python stylegan2_ewc.py --exp class_name --iter no_of_iterations --g_ckpt pre_trained_GAN_checkpoint --size 32 ```



### CELEBAHQ_Adaptation
To adapt to a certain feature in CelebA_HQ dataset
```python stylegan2_ewc_train.py --exp feature_name --iter no_pf_iterations --gan_ckpt path_to_pretrained_GAN ```



