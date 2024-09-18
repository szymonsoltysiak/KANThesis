import torch
from torchvision import datasets, transforms
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import numpy as np

test_dataset = datasets.GTSRB(root='./data', split='test', transform=None, download=True)

to_tensor = transforms.ToTensor()

def apply_blur(image, blur_radius=2):
    return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

def apply_darkness(image, factor=0.4):  
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

rain_transform = A.Compose([
    A.RandomRain(brightness_coefficient=1, drop_width=1, blur_value=2, p=1),
    ToTensorV2(),
])

snow_transform = A.Compose([
    A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=1),
    ToTensorV2(),
])

fog_transform = A.Compose([
    A.RandomFog(fog_coef_lower=0.7, fog_coef_upper=0.8, alpha_coef=0.1, p=1),
    ToTensorV2(),
])

sunflare_transform = A.Compose([
    A.RandomSunFlare(
        flare_roi=(0.2, 0.2, 0.8, 0.8), 
        angle_lower=0.5,                
        src_radius=16,                   
        p=1
    ),
    ToTensorV2(),
])

def apply_disturbance(image, disturbance_type):
    if disturbance_type == 'blur':
        disturbed_image = apply_blur(image)
    elif disturbance_type == 'dark':
        disturbed_image = apply_darkness(image)
    elif disturbance_type == 'rain':
        disturbed_image = rain_transform(image=np.array(image))['image']
        disturbed_image = transforms.ToPILImage()(disturbed_image)
    elif disturbance_type == 'snow':
        disturbed_image = snow_transform(image=np.array(image))['image']
        disturbed_image = transforms.ToPILImage()(disturbed_image) 
    elif disturbance_type == 'fog':
        disturbed_image = fog_transform(image=np.array(image))['image']
        disturbed_image = transforms.ToPILImage()(disturbed_image)
    elif disturbance_type == 'sunflare':
        disturbed_image = sunflare_transform(image=np.array(image))['image']
        disturbed_image = transforms.ToPILImage()(disturbed_image)
    
    return disturbed_image

def show_all_disturbances(disturbances):
    fig, axs = plt.subplots(len(disturbances), 2, figsize=(10, 10))

    for i, disturbance_type in enumerate(disturbances):
        idx = random.randint(0, len(test_dataset) - 1)
        image, _ = test_dataset[idx]
        
        disturbed_image = apply_disturbance(image, disturbance_type)

        axs[i, 0].imshow(image)
        axs[i, 0].set_title('Original Image')
        axs[i, 0].axis('off')

        axs[i, 1].imshow(disturbed_image)
        axs[i, 1].set_title(f'After {disturbance_type}')
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

disturbances = ['blur', 'dark', 'rain', 'snow', 'fog', 'sunflare']

show_all_disturbances(disturbances)
