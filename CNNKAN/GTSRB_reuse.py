import torch
import random
import torch.nn as nn
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from KANConvKANLinear import KKAN_Convolutional_Network
from modelutils import test
from datautils import transform, label_to_name, imshow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KKAN_Convolutional_Network(device = device)
model.load_state_dict(torch.load('gtsrb_cnnkan_model.pth'))

test_dataset = datasets.GTSRB(root='./data', split='test', transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

test_loss, accuracy, precision, recall, f1 = test(model, device, test_loader, criterion=nn.CrossEntropyLoss())

print(f'Test Loss: {test_loss:.4f}')
print(f'Accuracy: {accuracy:.2%}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

indices = random.sample(range(len(test_dataset)), 3)

for idx in indices:
    image, label = test_dataset[idx]
    output = model(image.unsqueeze(0)) 
    _, predicted = torch.max(output, 1)
    
    actual_label_name = label_to_name[label]
    predicted_label_name = label_to_name[predicted.item()]
    
    print(f"Actual Label: {actual_label_name}, Predicted Label: {predicted_label_name}")
    imshow(image)