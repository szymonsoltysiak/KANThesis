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
from datautils import transform
from disturbances import apply_blur, add_rain_effect, apply_brightness

# Custom transformation class to apply rain effect followed by resizing and normalization
class RainEffectTransform:
    def __init__(self, rain_percentage=0.0, size=(32, 32)):
        self.rain_percentage = rain_percentage
        self.resize = transforms.Resize(size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __call__(self, image):
        # Convert the image to a PIL image if it's not already
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        # Apply the rain effect
        disturbed_image = add_rain_effect(image, rain_percentage=self.rain_percentage)

        # Ensure the disturbed image is a PIL image
        if isinstance(disturbed_image, np.ndarray):
            disturbed_image = Image.fromarray(disturbed_image)

        # Resize, convert to tensor, and normalize
        return self.normalize(self.to_tensor(self.resize(disturbed_image)))

# Define the models to evaluate
models = {
    'CNN': CNNModel(),
    'CNNKan': CNNKan(),
    'KANConvLinear': KANConvLinear()
}

# Load model state dictionaries
model_paths = {
    'CNN': 'gtsrb_cnn_model.pth',
    'CNNKan': 'gtsrb_cnnkan_model.pth',
    'KANConvLinear': 'gtsrb_kanconvkanlinear_model.pth'
}

# Initialize an empty dictionary to store accuracies
accuracies_dict = {model_name: [] for model_name in models.keys()}

rain_percentages = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Evaluate each model for different rain percentages
for model_name, model in models.items():
    # Load the model weights
    try:
        model.load_state_dict(torch.load(model_paths[model_name], weights_only=False))
    except FileNotFoundError as e:
        print(f"Error loading model '{model_name}': {e}")
        sys.exit(1)

    model.eval()
    print(f'Loaded model: {model_name}')

    for rain_percentage in rain_percentages:
        # Define the transformation including the custom rain effect
        initial_transform = RainEffectTransform(rain_percentage=rain_percentage, size=(32, 32))

        # Load the test dataset with the transformation
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
        print(f'{model_name} - Rain Percentage: {rain_percentage*100:.0f}%, Test Accuracy: {accuracy:.2f}%')

# Plotting the results
for model_name, accuracies in accuracies_dict.items():
    plt.plot(rain_percentages, accuracies, marker='o', label=model_name)

plt.xlabel('Rain Percentage')
plt.ylabel('Test Accuracy (%)')
plt.title('Model Accuracy vs. Rain Effect Intensity')
plt.legend()
plt.grid()
plt.show()
