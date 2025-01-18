import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from disturbances import apply_blur, apply_brightness, add_rain_effect, apply_rotation

class BlurEffectTransform:
    def __init__(self, blur_radius=0.0, size=(32, 32)):
        self.blur_radius = blur_radius
        self.resize = transforms.Resize(size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        if self.blur_radius != 0:
            disturbed_image = apply_blur(image, blur_radius=self.blur_radius)
        else:
            disturbed_image = image

        if isinstance(disturbed_image, np.ndarray):
            disturbed_image = Image.fromarray(disturbed_image)

        return self.normalize(self.to_tensor(self.resize(disturbed_image)))
    
class RotationEffectTransform:
    def __init__(self, angle=0.0, size=(32, 32)):
        self.angle = angle
        self.resize = transforms.Resize(size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        if self.angle != 0:
            disturbed_image = apply_rotation(image, angle=self.angle)
        else:
            disturbed_image = image

        if isinstance(disturbed_image, np.ndarray):
            disturbed_image = Image.fromarray(disturbed_image)

        return self.normalize(self.to_tensor(self.resize(disturbed_image)))
    
class RainEffectTransform:
    def __init__(self, rain_percentage=0.0, size=(32, 32)):
        self.rain_percentage = rain_percentage
        self.resize = transforms.Resize(size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        if self.rain_percentage != 0:
            disturbed_image = add_rain_effect(image, rain_percentage=self.rain_percentage)
        else:
            disturbed_image = image

        if isinstance(disturbed_image, np.ndarray):
            disturbed_image = Image.fromarray(disturbed_image)

        return self.normalize(self.to_tensor(self.resize(disturbed_image)))
    
class BrightnessEffectTransform:
    def __init__(self, brightness_factor=0.0, size=(32, 32)):
        self.brightness_factor = brightness_factor
        self.resize = transforms.Resize(size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        if self.brightness_factor != 1:
            disturbed_image = apply_brightness(image, brightness_factor=self.brightness_factor)
        else:
            disturbed_image = image

        if isinstance(disturbed_image, np.ndarray):
            disturbed_image = Image.fromarray(disturbed_image)

        return self.normalize(self.to_tensor(self.resize(disturbed_image)))

class DualTransform:
    def __init__(self, disturbance_class, size=(32, 32), coeffs=None):
        self.disturbance_class = disturbance_class
        self.size = size
        self.coeffs = coeffs if coeffs is not None else [0] 
        self.base_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __call__(self, image):
        original = self.base_transform(image)

        disturbance_coeff = random.choice(self.coeffs)

        disturbed_transform = self.disturbance_class(disturbance_coeff, self.size)
        disturbed = disturbed_transform(image)

        return original, disturbed

class RandomDisturbanceTransform:
    def __init__(self, disturbances, disturbance_coeffs, size=(32, 32)):
        self.disturbances = disturbances
        self.disturbance_coeffs = disturbance_coeffs
        self.size = size
        self.base_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __call__(self, image):
        disturbance_type = random.choice(list(self.disturbances.keys()))
        disturbance_class = self.disturbances[disturbance_type]

        disturbance_coeff = random.choice(self.disturbance_coeffs[disturbance_type])

        disturbed_transform = disturbance_class(disturbance_coeff, self.size)
        disturbed_image = disturbed_transform(image)

        original_image = self.base_transform(image)

        return original_image, disturbed_image
