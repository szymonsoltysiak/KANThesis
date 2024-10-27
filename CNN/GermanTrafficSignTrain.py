import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import sys
import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from CNNmodel import CNNModel
from CNNKanInSeries import CNNKan
from KANConvModel.KANConvKANLinear import KANConvLinear 
from datautils import transform

mode = 'CNN' # CNN or CNNKan or KANConvLinear

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

trn_losses = []
val_losses = []
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
        trn_losses.append(loss.item())
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
    val_losses.append(val_loss / len(val_loader))
    val_acc.append(100 * correct / total)

fig, axs = plt.subplots(3, 1, figsize=(10, 15))

axs[0].plot(range(len(trn_losses)), trn_losses, label='Training Loss', color='blue')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training Loss over Steps')
axs[0].legend()

axs[1].plot(range(num_epochs), val_losses, label='Validation Loss', color='orange')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')
axs[1].set_title('Validation Loss over Epochs')
axs[1].legend()

axs[2].plot(range(num_epochs), val_acc, label='Validation Accuracy', color='green')
axs[2].set_xlabel('Epochs')
axs[2].set_ylabel('Accuracy (%)')
axs[2].set_title('Validation Accuracy over Epochs')
axs[2].legend()

plt.tight_layout()
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