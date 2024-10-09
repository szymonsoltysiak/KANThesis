from torchvision import datasets, transforms
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2

test_dataset = datasets.GTSRB(root='./data', split='test', transform=None, download=True)

to_tensor = transforms.ToTensor()

def apply_blur(image, blur_radius=2):
    return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

def add_rain_effect(image, rain_percentage=0.01, drop_length=5, drop_thickness=1, intensity=0.6, blur=2):
    if isinstance(image, Image.Image):
        image = np.array(image)

    rainy_image = image.copy()

    
    height, width, _ = image.shape
    
    num_pixels = height * width
    num_drops = int(rain_percentage * num_pixels / (drop_length*drop_thickness*blur))  
    
    rain_layer = np.zeros((height, width), dtype=np.uint8)
    
    for _ in range(num_drops):
        start_x = np.random.randint(0, width)
        start_y = np.random.randint(0, height)
        
        end_x = start_x + np.random.randint(-2, 2) 
        end_y = start_y + np.random.randint(drop_length // 2, drop_length)  
        
        cv2.line(rain_layer, (start_x, start_y), (end_x, end_y), 255, thickness=drop_thickness)
    
    rain_layer = cv2.blur(rain_layer, (blur, blur))
    
    rain_layer_colored = cv2.cvtColor(rain_layer, cv2.COLOR_GRAY2BGR)
    
    alpha = intensity 
    beta = 1 - intensity + 0.4
    gamma = 10

    rainy_image = cv2.addWeighted(rain_layer_colored, alpha, rainy_image, beta, gamma)
    
    return rainy_image


def apply_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def show_rain_with_different_percentages(rain_percentages):
    num_images = len(rain_percentages)
    fig, axs = plt.subplots(1, num_images + 1, figsize=(5 * (num_images + 1), 5))  

    idx = random.randint(0, len(test_dataset) - 1)
    original_image, _ = test_dataset[idx]
    
    if not isinstance(original_image, Image.Image):
        original_image = transforms.ToPILImage()(original_image)

    axs[0].imshow(original_image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    for i, rain_percentage in enumerate(rain_percentages):
        disturbed_image = add_rain_effect(original_image, rain_percentage=rain_percentage)

        axs[i + 1].imshow(disturbed_image)
        axs[i + 1].set_title(f'Rain {rain_percentage * 100}%')
        axs[i + 1].axis('off')

    plt.tight_layout()
    plt.show()


def show_brightness_variations(brightness_factors):
    num_images = len(brightness_factors)
    mid_index = num_images // 2 
    
    fig, axs = plt.subplots(1, num_images, figsize=(5 * num_images, 5))  

    idx = random.randint(0, len(test_dataset) - 1)
    original_image, _ = test_dataset[idx]

    if not isinstance(original_image, Image.Image):
        original_image = transforms.ToPILImage()(original_image)

    for i, factor in enumerate(brightness_factors):
        if i == mid_index:
            axs[i].imshow(original_image)
            axs[i].set_title(f'Original')
        else:
            adjusted_image = apply_brightness(original_image, factor)
            if factor > 1:
                axs[i].set_title(f'Brighter {factor:.2f}')
            else:
                axs[i].set_title(f'Darker {factor:.2f}')
            axs[i].imshow(adjusted_image)
        
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

def show_all(rain_percentages, brightness_factors):
    num_rain_images = len(rain_percentages)
    num_brightness_images = len(brightness_factors)
    
    total_images = num_rain_images + num_brightness_images + 1 
    fig, axs = plt.subplots(2, total_images // 2, figsize=(5 * (total_images // 2), 10))  
    
    idx = random.randint(0, len(test_dataset) - 1)
    original_image, _ = test_dataset[idx]
    
    if not isinstance(original_image, Image.Image):
        original_image = transforms.ToPILImage()(original_image)

    mid_rain = num_rain_images // 2
    mid_brightness = num_brightness_images // 2

    axs[0, mid_rain].imshow(original_image)
    axs[0, mid_rain].set_title('Original Image (Rain)')
    axs[0, mid_rain].axis('off')

    for i, rain_percentage in enumerate(rain_percentages):
        if i == len(rain_percentages) -1 :
            disturbed_image = apply_blur(original_image, blur_radius=2)
            axs[0, i].set_title(f'Blur')

        else:      
            disturbed_image = add_rain_effect(original_image, rain_percentage=rain_percentage)
            axs[0, i].set_title(f'Rain {rain_percentage * 100}%')
        
        axs[0, i].axis('off')
        axs[0, i].imshow(disturbed_image)

    axs[1, mid_brightness].imshow(original_image)
    axs[1, mid_brightness].set_title('Original Image (Brightness)')
    axs[1, mid_brightness].axis('off')

    for i, factor in enumerate(brightness_factors):
        adjusted_image = apply_brightness(original_image, factor)
        axs[1, i].imshow(adjusted_image)
        if factor > 1:
            axs[1, i].set_title(f'Brighter {factor:.2f}')
        else:
            axs[1, i].set_title(f'Darker {factor:.2f}')
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()

rain_percentages = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1]

brightness_factors = [2.0, 1.5, 1.2, 1.0, 0.8, 0.5, 0.3]

show_all(rain_percentages, brightness_factors)
