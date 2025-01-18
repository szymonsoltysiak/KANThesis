import sys
import torch
import json
import torchvision.datasets as datasets

from CNNmodel import CNNModel
from CNNKanInSeries import CNNKan 
from KANConvModel.KANConvKANLinear import KANConvLinear

from torch.utils.data import DataLoader
from datautils import transform
from DisturbanceTransformations import BlurEffectTransform, RotationEffectTransform, RainEffectTransform, BrightnessEffectTransform

disturbance_mode = "blur" # blur, rotation, rain, brightness
trained_with_disturbances = True 

disturbances = {
    'blur': BlurEffectTransform,
    'rotation': RotationEffectTransform,
    'rain': RainEffectTransform,
    'brightness': BrightnessEffectTransform
}

disturbance_coeffs = {
    'blur': [0,1,2,3,4,5,6,8,10,12,16,20,24],
    'rotation': [-180, -165, -150, -135, -120, -105, -90, -75, -60, -45,-40,-35, -30,-25,-20, -15,-10,-5, 0,5,10, 15,20,25, 30,35,40, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180],
    'rain': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'brightness': [20.0, 15.0, 10.0, 5.0, 3.0, 2.0, 1.5, 1.25, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
}

disturbance_transform = disturbances[disturbance_mode]
coeffs = disturbance_coeffs[disturbance_mode]

models = {
    'CNN': CNNModel(),
    'CNNKAN': CNNKan(),
    'CKAN': KANConvLinear()
}

model_paths = {
    'CNN': 'gtsrb_cnn_model.pth',
    'CNNKAN': 'gtsrb_cnnkan_model.pth',
    'CKAN': 'gtsrb_ckan_model.pth'
}

accuracies_dict = {model_name: [] for model_name in models.keys()}

for model_name, model in models.items():
    try:
        path = 'models/'
        if trained_with_disturbances:
            path += (disturbance_mode + '_')
        path += model_paths[model_name]
        model.load_state_dict(torch.load(path, weights_only=False))
    except FileNotFoundError as e:
        print(f"Error loading model '{model_name}': {e}")
        sys.exit(1)

    model.eval()
    print(f'Loaded model: {model_name}')

    for coeff in coeffs:
        initial_transform = disturbance_transform(coeff, (32, 32))

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
        print(f'{model_name} - {disturbance_mode} coeff: {coeff:.2f}, Test Accuracy: {accuracy:.2f}%')


output_data = {
    "coeff": coeffs,
    **{model_name: accuracies for model_name, accuracies in accuracies_dict.items()}
}

file_name = f'accuracies_{disturbance_mode}.json'

with open(file_name, 'w') as json_file:
    json.dump(output_data, json_file)
