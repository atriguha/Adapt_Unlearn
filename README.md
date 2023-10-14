
# Adapt then Unlearn: Exploiting Parameter Space Semantics for Unlearning in Generative Adversarial Networks




## Table of Contents


- [Prerequisites](#prerequisites)
- [Classifiers](#classifiers)
  - [MNIST](#mnist)
  - [CelebA](#celeba)
- [Adaptation](#adaptation)
  
- [Unlearning](#unlearning)




## Prerequisites
Required checkpoints for 
1. Pre Trained Stylegan2 on CELEBA-HQ
2. Pre Trained Stylegan2 on MNIST
3. Pre Trained Classifiers


## Classifiers

### MNIST
Checkpoints And Model for MNIST Classifier is available in this repository itself. We have used the checkpoints provided in [this](https://github.com/csinva/gan-vae-pretrained-pytorch/tree/master/mnist_classifier) repository

### CelebA

For training and implementation of a classifier for classifying facial attributes in CelebA dataset, we have referred to [this](https://github.com/rgkannan676/Recognition-and-Classification-of-Facial-Attributes/tree/main) repository. 


## Adaptation

The implementation for Adaptation is done seperately for MNIST and CELEBA-HQ . They can be found in ADAPT_MNIST and ADAPT_CELEBA folders respectively.

* To adapt to a certain class in MNIST

```bash
 python stylegan2_ewc.py --exp class_name --iter no_of_iterations --g_ckpt pre_trained_GAN_checkpoint --size 32 
 ```



* To adapt to a certain feature in CelebA_HQ dataset

```bash
python stylegan2_ewc_train.py --exp feature_name --iter no_pf_iterations --gan_ckpt path_to_pretrained_GAN 
```



## Unlearning

* For class level unlearning on MNIST

```bash
python train_different_losses.py --expt class_name \
 --list_of_models path_to_model1 \
     path_to_model2 \
     path_to_model3 \
     path_to_model4 \
     path_to_model5 \
    --size 32 --ckpt path_to_pre_trained_GAN --iter no_of_iterations --repel --gamma value_of_constant --loss_type type_of_loss_function

```
* For feature level unlearning on CELEBA_HQ
```bash
python unlearn_main.py --expt class_name \
 --list_of_models path_to_model1 \
     path_to_model2 \
     path_to_model3 \
     path_to_model4 \
     path_to_model5 \
     --ckpt path_to_pre_trained_GAN --iter no_of_iterations --repel_loss --gamma value_of_constant --loss_type type_of_loss_function


```
### FEATURE LIST OF CELEBA-HQ
| Attribute            | Index | Attribute            | Index | Attribute         | Index | Attribute           | Index | Attribute          | Index |
|----------------------|-------|----------------------|-------|-------------------|-------|---------------------|-------|--------------------|-------|
| **Attributes**       | **Index** | **Attributes**       | **Index** | **Attributes**    | **Index** | **Attributes**      | **Index** | **Attributes**    | **Index** |
|----------------------|-------|----------------------|-------|-------------------|-------|---------------------|-------|--------------------|-------|
| 5_o_Clock_Shadow     | 0     | Arched_Eyebrows      | 1     | Attractive        | 2     | Bags_Under_Eyes    | 3     | Bald               | 4     |
| Bangs                | 5     | Big_Lips             | 6     | Big_Nose           | 7     | Black_Hair         | 8     | Blond_Hair         | 9     |
| Blurry               | 10    | Brown_Hair           | 11    | Bushy_Eyebrows     | 12    | Chubby             | 13    | Double_Chin        | 14    |
| Eyeglasses           | 15    | Goatee               | 16    | Gray_Hair          | 17    | HeavyMakeup        | 18    | HighCheekbones     | 19    |
| Male                 | 20    | MouthSlightlyOpen    | 21    | Mustache           | 22    | NarrowEyes         | 23    | NoBeard            | 24    |
| OvalFace             | 25    | PaleSkin             | 26    | PointyNose         | 27    | RecedingHairline   | 28    | RosyCheeks         | 29    |
| Sideburns            | 30    | Smiling              | 31    | StraightHair       | 32    | WavyHair           | 33    | WearingEarrings    | 34    |
| WearingHat           | 35    | WearingLipstick      | 36    | WearingNecklace     | 37    | WearingNecktie     | 38    | Young              | 39    |




### NOTE REGARDING THE TYPE OF REPULSION LOSS FUNCTION
* For $\{L}_{repulsion}^{{EL2}}$, use ```args.loss_type="exp"```

* For $\mathcal{L}_{repulsion}^{{NL2}}$, use ```args.loss_type="l2"```
* For $\mathcal{L}_{repulsion}^{{IL2}}$, use ```args.loss_type="reci"```





