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

def get_average_color(image):
    np_image = np.array(image)
    avg_color_per_row = np.average(np_image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return tuple(avg_color.astype(int))

def zoom_image(image, zoom_factor=1.5):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    width, height = image.size
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    zoomed_image = image.crop((left, top, right, bottom)).resize((width, height), Image.Resampling.LANCZOS)
    return zoomed_image

def apply_rotation(image, angle):
    rotated_image = image.convert("RGBA").rotate(angle, expand=True)
    average_color = get_average_color(image)
    background = Image.new("RGBA", rotated_image.size, (*average_color, 255))  
    background.paste(rotated_image, (0, 0), rotated_image)
    background.convert("RGB")
    angle_radians = np.radians(angle)
    return zoom_image(background, min(abs(np.sin(angle_radians)), abs(np.cos(angle_radians))) + 1)

def show_all(rain_percentages, brightness_factors, rotation_angles):
    num_rain_images = len(rain_percentages)
    num_brightness_images = len(brightness_factors)
    num_rotation_images = len(rotation_angles)
    
    total_images = num_rain_images + num_brightness_images + num_rotation_images + 1 
    fig, axs = plt.subplots(3, total_images // 3, figsize=(5 * (total_images // 3), 15))  
    
    idx = random.randint(0, len(test_dataset) - 1)
    original_image, _ = test_dataset[idx]
    
    if not isinstance(original_image, Image.Image):
        original_image = transforms.ToPILImage()(original_image)

    mid_rain = num_rain_images // 2
    mid_brightness = num_brightness_images // 2
    mid_rotation = num_rotation_images // 2

    axs[0, mid_rain].imshow(original_image)
    axs[0, mid_rain].set_title('Original Image (Rain)')
    axs[0, mid_rain].axis('off')

    for i, rain_percentage in enumerate(rain_percentages):
        if i == len(rain_percentages) - 1:
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

    axs[2, mid_rotation].imshow(original_image)
    axs[2, mid_rotation].set_title('Original Image (Rotation)')
    axs[2, mid_rotation].axis('off')

    for i, angle in enumerate(rotation_angles):
        rotated_image = apply_rotation(original_image, angle)
        axs[2, i].imshow(rotated_image)
        axs[2, i].set_title(f'Rotate {angle}°')
        axs[2, i].axis('off')

    plt.tight_layout()
    plt.show()

rain_percentages = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1]
brightness_factors = [2.0, 1.5, 1.2, 1.0, 0.8, 0.5, 0.3]
rotation_angles = [-90, -45, -20, 0, 20, 45, 90]

#show_all(rain_percentages, brightness_factors, rotation_angles)
