import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

from CNNmodel import CNNModel
from CNNKanInSeries import CNNKan 
from KANConvModel.KANConvKANLinear import KANConvLinear

from torch.utils.data import DataLoader
from disturbances import apply_brightness

class BrightnessEffectTransform:
    def __init__(self, brightness_factor=0.0, size=(32, 32)):
        self.brightness_factor = brightness_factor
        self.resize = transforms.Resize(size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        disturbed_image = apply_brightness(image, brightness_factor=self.brightness_factor)

        if isinstance(disturbed_image, np.ndarray):
            disturbed_image = Image.fromarray(disturbed_image)

        return self.normalize(self.to_tensor(self.resize(disturbed_image)))

models = {
    'CNN': CNNModel(),
    'CNNKan': CNNKan(),
    'KANConvLinear': KANConvLinear()
}

model_paths = {
    'CNN': 'gtsrb_cnn_model.pth',
    'CNNKan': 'gtsrb_cnnkan_model.pth',
    'KANConvLinear': 'gtsrb_kanconvkanlinear_model.pth'
}

accuracies_dict = {model_name: [] for model_name in models.keys()}

brightness_factors = [20.0,15.0,10.0,5.0, 3.0, 2.0, 1.5,1.25, 1.0,0.8,0.6,0.5,0.4, 0.3, 0.2, 0.1, 0.0]

for model_name, model in models.items():
    try:
        model.load_state_dict(torch.load(model_paths[model_name], weights_only=False))
    except FileNotFoundError as e:
        print(f"Error loading model '{model_name}': {e}")
        sys.exit(1)

    model.eval()
    print(f'Loaded model: {model_name}')

    for brightness_factor in brightness_factors:
        initial_transform = BrightnessEffectTransform(brightness_factor=brightness_factor, size=(32, 32))

        test_dataset = datasets.GTSRB(root='./data', split='test', transform=initial_transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracies_dict[model_name].append(accuracy)
        print(f'{model_name} - Brightness_factor: {brightness_factor:.2f}, Test Accuracy: {accuracy:.2f}%')

for model_name, accuracies in accuracies_dict.items():
    plt.plot(brightness_factors, accuracies, marker='o', label=model_name)

plt.xlabel('Brightness factor')
plt.ylabel('Test Accuracy (%)')
plt.title('Model Accuracy vs. Brightness factor')
plt.legend()
plt.grid()
plt.xscale('symlog')
plt.show()
