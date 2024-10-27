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

mode = 'CNNKan' # CNN or CNNKan or KANConvLinear

train_dataset = datasets.GTSRB(root='./data', split='train', transform=transform, download=True)
test_dataset = datasets.GTSRB(root='./data', split='test', transform=transform, download=True)

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

if mode == 'CNNKan':
    model = CNNKan()
elif mode == 'KANConvLinear':
    model = KANConvLinear()
elif mode == 'CNN':
    model = CNNModel()
else:
    raise ValueError('Invalid mode')
    sys.exit()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

val_loss = []
val_acc = []

num_epochs = 10
for epoch in range(num_epochs):
    print(f'Epoch [{epoch + 1}/{num_epochs}] started {datetime.datetime.now()}')
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    scheduler.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f} {datetime.datetime.now()}')


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
    val_loss.append(val_loss / len(val_loader))
    val_acc.append(100 * correct / total)

    import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(val_loss)+1), val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Validation Loss over Epochs')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(val_acc)+1), val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy over Epochs')
plt.legend()
plt.show()

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
    'CNN': 'gtsrb_cnn_model.pth',
    'CNNKan': 'gtsrb_cnnkan_model.pth',
    'KANConvLinear': 'gtsrb_kanconvkanlinear_model.pth'
}
torch.save(model.state_dict(), model_paths[mode])