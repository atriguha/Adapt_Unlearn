
# Adapt_Unlearn

Brief description or introduction of your project.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  
- [Adaptation](#adaptation)
    - [MNIST_Adaptation ](#adapt_mnist)
    - [CELEBAHQ_Adaptation ](#adapt_celebahq)
    

- [Unlearning](#unlearning)
    - [MNIST_Unlearning]
    - [CELEBAHQ_Unlearning]
- [Contributing](#contributing)
- [License](#license)

## Getting Started(#getting-started)

Explain what someone needs to do to get started with your project.

### Prerequisites(#prerequisites)
Required checkpoints for 
1. Pre Trained Stylegan2 on CELEBA-HQ
2. Pre Trained Stylegan2 on MNIST
3. Pre Trained Classifiers

## Adaptation(#adaptation)

The implementation for Adaptation is done seperately for MNIST and CELEBA-HQ . They can be found in ADAPT_MNIST and ADAPT_CELEBA folders respectively.

### MNIST_Adaptation(#adapt_mnist)
To adapt to a certain class in MNIST
``` python stylegan2_ewc.py --exp class_name --iter no_of_iterations --g_ckpt pre_trained_GAN_checkpoint --size 32 ```



### CELEBAHQ_Adaptation(#adapt_celebahq)
To adapt to a certain feature in CelebA_HQ dataset
```python stylegan2_ewc_train.py --exp feature_name --iter no_pf_iterations --gan_ckpt path_to_pretrained_GAN ```



## Unlearning(#unlearning)

### MNIST_Unlearning
For class level unlearning on MNIST
```bash
python train_different_losses.py --expt class_name \
 --list_of_models path_to_model1 \
     path_to_model2 \
     path_to_model3 \
     path_to_model4 \
     path_to_model5 \
    --size 32 --ckpt path_to_pre_trained_GAN --iter no_of_iterations --repel --gamma value_of_constant --loss_type type_of_loss_function


```
### CELEBAHQ_Unlearning
For feature level unlearning on CELEBA_HQ
```bash
python unlearn_main.py --expt class_name \
 --list_of_models path_to_model1 \
     path_to_model2 \
     path_to_model3 \
     path_to_model4 \
     path_to_model5 \
     --ckpt path_to_pre_trained_GAN --iter no_of_iterations --repel_loss --gamma value_of_constant --loss_type type_of_loss_function


```
This sentence uses `$` delimiters to show math inline:  $\sqrt{3x-1}+(1+x)^2$

### NOTE REGARDING THE TYPE OF LOSS FUNCTION
* For $\{L}_{repulsion}^{{EL2}}$, use ```args.loss_type="exp"```

* For $\mathcal{L}_{repulsion}^{{NL2}}$, use ```args.loss_type="l2"```
* For $\mathcal{L}_{repulsion}^{{IL2}}$, use ```args.loss_type="reci"```





