import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from KAN import KANet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

class MNISTDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image.view(-1), label

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
valset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

valloader = DataLoader(MNISTDataset(valset), batch_size=64, shuffle=False)

model = KANet([28 * 28, 64, 10])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model_load_path = 'kanet_minst.pth'
model.load_state_dict(torch.load(model_load_path))

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in tqdm(valloader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

accuracy = accuracy_score(all_labels, all_predictions)
print(f"Accuracy: {accuracy:.4f}")

cm = confusion_matrix(all_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))

disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

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
