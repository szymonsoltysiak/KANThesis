import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import sys
import datetime
from torch.utils.data import DataLoader, random_split
from CNNmodel import CNNModel
from CNNKanInSeries import CNNKan
from KANConvModel.KANConvKANLinear import KANConvLinear 
from datautils import transform
from DisturbanceTransformations import BlurEffectTransform, RotationEffectTransform, RainEffectTransform, BrightnessEffectTransform, RandomDisturbanceTransform

mode = 'CNN'  # CNN or CNNKAN or CKAN

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

train_transform = RandomDisturbanceTransform(disturbances, disturbance_coeffs, (32,32))
train_dataset = datasets.GTSRB(root='./data', split='train', transform=train_transform, download=True)
test_dataset = datasets.GTSRB(root='./data', split='test', transform=transform, download=True)

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

def collate_fn_augmented(batch):
    original_images = torch.cat([b[0][0].unsqueeze(0) for b in batch], dim=0) 
    disturbed_images = torch.cat([b[0][1].unsqueeze(0) for b in batch], dim=0) 
    labels = torch.cat([torch.tensor(b[1]).unsqueeze(0) for b in batch], dim=0)  
    
    all_images = torch.cat([original_images, disturbed_images], dim=0) 
    all_labels = torch.cat([labels, labels], dim=0)
    
    return all_images, all_labels

def collate_fn_orginal(batch):
    original_images = torch.cat([b[0][0].unsqueeze(0) for b in batch], dim=0) 
    labels = torch.cat([torch.tensor(b[1]).unsqueeze(0) for b in batch], dim=0)  
        
    return original_images, labels

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_augmented)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_orginal)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

if mode == 'CNNKAN':
    model = CNNKan()
elif mode == 'CKAN':
    model = KANConvLinear()
elif mode == 'CNN':
    model = CNNModel()
else:
    raise ValueError('Invalid mode')
    sys.exit()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

trn_losses = []
val_losses = []
val_acc = []

num_epochs = 10
for epoch in range(num_epochs):
    print(f'Epoch [{epoch + 1}/{num_epochs}] started {datetime.datetime.now()}')
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        batch_size = images.size(0) // 2
        original_images = images[:batch_size]
        disturbed_images = images[batch_size:]
        all_images = torch.cat([original_images, disturbed_images], dim=0)

        all_labels = torch.cat([labels[:batch_size], labels[:batch_size]], dim=0)

        optimizer.zero_grad()
        outputs = model(all_images)
        loss = criterion(outputs, all_labels)
        trn_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_loader):.4f}')

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Loss: {val_loss / len(val_loader):.4f}')
    print(f'Validation Accuracy: {100 * correct / total:.2f}%')
    val_losses.append(val_loss / len(val_loader))
    val_acc.append(100 * correct / total)


model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Loss: {test_loss / len(test_loader):.4f}')
print(f'Test Accuracy: {100 * correct / total:.2f}%')

model_paths = {
    'CNN': '_gtsrb_cnn_model.pth',
    'CNNKAN': '_gtsrb_cnnkan_model.pth',
    'CKAN': '_gtsrb_ckan_model.pth'
}

file_name = f'mix_gtsrb_{mode.lower()}_model.pth'

torch.save(model.state_dict(), file_name)

import json

metrics = {
    "training_losses": trn_losses,
    "validation_losses": val_losses,
    "validation_accuracies": val_acc
}

metrics_file = f"mix_{mode.lower()}_metrics.json"
with open(metrics_file, "w") as f:
    json.dump(metrics, f)

print(f"Metrics saved to {metrics_file}")
