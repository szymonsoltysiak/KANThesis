import sys
import torch
import random
import torchvision.datasets as datasets

from CNNmodel import CNNModel
from CNNKanInSeries import CNNKan 
from KANConvModel.KANConvKANLinear import KANConvLinear

from torch.utils.data import DataLoader
from datautils import transform, label_to_name, imshow

mode = 'KANConvLinear' # CNN or CNNKan or KANConvLinear
evaluate = True
show_examples = True

if mode == 'CNNKan':
    model = CNNKan()
elif mode == 'KANConvLinear':
    model = KANConvLinear()
elif mode == 'CNN':
    model = CNNModel()
else:
    raise ValueError('Invalid mode')
    sys.exit()

if mode == 'CNNKan':
    model.load_state_dict(torch.load('gtsrb_cnnkan_model.pth', weights_only=False))
elif mode == 'KANConvLinear':
    model.load_state_dict(torch.load('gtsrb_kanconvkanlinear_model.pth', weights_only=False))
if mode == 'CNN':  
    model.load_state_dict(torch.load('gtsrb_cnn_model.pth', weights_only=False))

model.eval()

test_dataset = datasets.GTSRB(root='./data', split='test', transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

if evaluate:
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

if show_examples:
    indices = random.sample(range(len(test_dataset)), 3)

    for idx in indices:
        image, label = test_dataset[idx]
        output = model(image.unsqueeze(0)) 
        _, predicted = torch.max(output, 1)
        
        actual_label_name = label_to_name[label]
        predicted_label_name = label_to_name[predicted.item()]
        
        print(f"Actual Label: {actual_label_name}, Predicted Label: {predicted_label_name}")
        imshow(image)