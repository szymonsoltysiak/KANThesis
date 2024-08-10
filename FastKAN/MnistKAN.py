import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from KAN import KANet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
valset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

model = KANet([28 * 28, 64, 10])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3)

criterion = nn.CrossEntropyLoss()

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True)

model_save_path = 'kanet_minst.pth'

for epoch in range(5):
    model.train()
    total_loss = 0
    total_accuracy = 0
    with tqdm(trainloader) as pbar:
        for images, labels in pbar:
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            accuracy = (output.argmax(dim=1) == labels).float().mean()
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item())
    total_loss /= len(trainloader)
    total_accuracy /= len(trainloader)

    model.eval()
    val_loss = 0
    val_accuracy = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in valloader:
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)
            output = model(images)
            val_loss += criterion(output, labels).item()
            val_accuracy += (output.argmax(dim=1) == labels).float().mean().item()
            all_preds.extend(output.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_loss /= len(valloader)
    val_accuracy /= len(valloader)

    scheduler.step(val_loss)

    print(f"Epoch {epoch + 1}, Train Loss: {total_loss}, Train Accuracy: {total_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

torch.save(model.state_dict(), model_save_path)

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
disp.plot()
plt.title(f'Confusion Matrix - After Training')
plt.show()

print("Training complete and model saved")

indices = np.random.choice(len(valset), 5, replace=False)
fig, axs = plt.subplots(1, 5, figsize=(15, 3))

for i, idx in enumerate(indices):
    image, true_label = valset[idx]
    image = image.view(-1).to(device)
    
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        predicted_label = output.argmax(dim=1).item()
    
    image = image.view(28, 28).cpu().numpy()
    image = (image * 0.5) + 0.5  
    
    axs[i].imshow(image, cmap='gray')
    axs[i].set_title(f"True: {true_label}\nPred: {predicted_label}")
    axs[i].axis('off')

plt.show()