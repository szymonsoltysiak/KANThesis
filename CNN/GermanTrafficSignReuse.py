import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import random
import sys
from CNNmodel import CNNModel
from CNNKanInSeries import CNNKan 
from datautils import transform, label_to_name, imshow

mode = 'CNNKan' # CNN or CNNKan

if mode == 'CNNKan':
    model = CNNKan()
elif mode == 'CNN':
    model = CNNModel()
else:
    raise ValueError('Invalid mode')
    sys.exit()

if mode == 'CNNKan':
    model.load_state_dict(torch.load('gtsrb_cnnkan_model.pth'))
if mode == 'CNN':  
    model.load_state_dict(torch.load('gtsrb_cnn_model.pth'))

model.eval()

test_dataset = datasets.GTSRB(root='./data', split='test', transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

indices = random.sample(range(len(test_dataset)), 3)

for idx in indices:
    image, label = test_dataset[idx]
    output = model(image.unsqueeze(0)) 
    _, predicted = torch.max(output, 1)
    
    actual_label_name = label_to_name[label]
    predicted_label_name = label_to_name[predicted.item()]
    
    print(f"Actual Label: {actual_label_name}, Predicted Label: {predicted_label_name}")
    imshow(image)