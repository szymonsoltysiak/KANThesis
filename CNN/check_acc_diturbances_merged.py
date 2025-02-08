import sys
import torch
import json
import torchvision.datasets as datasets

from CNNmodel import CNNModel
from CNNKanInSeries import CNNKan 
from KANConvModel.KANConvKANLinear import KANConvLinear

from torch.utils.data import DataLoader
from datautils import transform
from DisturbanceTransformations import RandomDisturbancesMergedTransform

disturbance_transform = RandomDisturbancesMergedTransform(size=(32, 32))
base_transform = transform

models = {
    'CNN': CNNModel(),
    'CNNKAN': CNNKan(),
    'CKAN': KANConvLinear(),
    'CNN_mix': CNNModel(),
    'CNNKAN_mix': CNNKan(),
    'CKAN_mix': KANConvLinear()
}

model_paths = {
    'CNN': 'gtsrb_cnn_model.pth',
    'CNNKAN': 'gtsrb_cnnkan_model.pth',
    'CKAN': 'gtsrb_ckan_model.pth',
    'CNN_mix': 'mix_gtsrb_cnn_model.pth',
    'CNNKAN_mix': 'mix_gtsrb_cnnkan_model.pth',
    'CKAN_mix': 'mix_gtsrb_ckan_model.pth'
}

accuracies_dict = {model_name: [] for model_name in models.keys()}

for model_name, model in models.items():
    try:
        path = 'models/'
        path += model_paths[model_name]
        model.load_state_dict(torch.load(path, weights_only=False))
    except FileNotFoundError as e:
        print(f"Error loading model '{model_name}': {e}")
        sys.exit(1)

    model.eval()
    print(f'Loaded model: {model_name}')

    initial_transform = disturbance_transform

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
    print(f'{model_name}, Test Accuracy: {accuracy:.2f}%')


output_data = {
    **{model_name: accuracies for model_name, accuracies in accuracies_dict.items()}
}

file_name = f'accuracies_no_disturbances_merged.json'

with open(file_name, 'w') as json_file:
    json.dump(output_data, json_file)
