import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from KANConvKANLinear import KKAN_Convolutional_Network
from modelutils import train_and_test_models
from datautils import transform

train_dataset = datasets.GTSRB(root='./data', split='train', transform=transform, download=True)
test_dataset = datasets.GTSRB(root='./data', split='test', transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = KKAN_Convolutional_Network(device = device)
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
criterion = nn.CrossEntropyLoss()

all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1 = train_and_test_models(model, device, train_loader, test_loader, optimizer, criterion, epochs=10, scheduler=scheduler)

torch.save(model.state_dict(), 'gtsrb_cnnkan_model.pth')

print(f'Final Train Loss: {all_train_loss[-1]:.6f}')
print(f'Final Test Loss: {all_test_loss[-1]:.4f}')
print(f'Final Test Accuracy: {all_test_accuracy[-1]:.2%}')
print(f'Final Test Precision: {all_test_precision[-1]:.2f}')
print(f'Final Test Recall: {all_test_recall[-1]:.2f}')
print(f'Final Test F1 Score: {all_test_f1[-1]:.2f}')