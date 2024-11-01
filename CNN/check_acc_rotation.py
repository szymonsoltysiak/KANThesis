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
from disturbances import apply_rotation

class RotationEffectTransform:
    def __init__(self, angle=0.0, size=(32, 32)):
        self.angle = angle
        self.resize = transforms.Resize(size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        disturbed_image = apply_rotation(image, angle=self.angle)

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

angles = [-180, -165, -150, -135, -120, -105, -90, -75, -60, -45,-40,-35, -30,-25,-20, -15,-10,-5, 0,5,10, 15,20,25, 30,35,40, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]


for model_name, model in models.items():
    try:
        model.load_state_dict(torch.load(model_paths[model_name], weights_only=False))
    except FileNotFoundError as e:
        print(f"Error loading model '{model_name}': {e}")
        sys.exit(1)

    model.eval()
    print(f'Loaded model: {model_name}')

    for angle in angles:
        initial_transform = RotationEffectTransform(angle=angle, size=(32, 32))

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
        print(f'{model_name} - Angle: {angle:.2f}, Test Accuracy: {accuracy:.2f}%')

for model_name, accuracies in accuracies_dict.items():
    plt.plot(angles, accuracies, marker='o', label=model_name)

plt.xlabel('Angle of rotation')
plt.ylabel('Test Accuracy (%)')
plt.title('Model Accuracy vs. Angle of rotation')
plt.legend()
plt.grid()
plt.show()
