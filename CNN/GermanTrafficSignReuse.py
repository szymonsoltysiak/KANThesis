import sys
import torch
import random
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import seaborn as sns  
from sklearn.metrics import confusion_matrix  

from CNNmodel import CNNModel
from CNNKanInSeries import CNNKan 
from KANConvModel.KANConvKANLinear import KANConvLinear

from torch.utils.data import DataLoader
from datautils import transform, label_to_name

mode = 'KANConvLinear' # CNN or CNNKan or KANConvLinear
evaluate = False
show_examples = False
show_confusion_matrix = False
print_params = True

if mode == 'CNNKan':
    model = CNNKan()
elif mode == 'KANConvLinear':
    model = KANConvLinear()
elif mode == 'CNN':
    model = CNNModel()
else:
    raise ValueError('Invalid mode')
    sys.exit()

model_paths = {
    'CNN': 'gtsrb_cnn_model.pth',
    'CNNKan': 'gtsrb_cnnkan_model.pth',
    'KANConvLinear': 'gtsrb_kanconvkanlinear_model.pth'
}
model.load_state_dict(torch.load(model_paths[mode], weights_only=False))

model.eval()

if print_params:
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'Layer: {name} | Number of parameters: {param.numel()}')

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of trainable parameters: {trainable_params}')

test_dataset = datasets.GTSRB(root='./data', split='test', transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

all_preds = []
all_labels = []

if evaluate:
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

if show_examples:
    num_examples = 5
    indices = random.sample(range(len(test_dataset)), num_examples)
    fig, axs = plt.subplots(1, num_examples, figsize=(10, 3))

    for i, idx in enumerate(indices):
        image, label = test_dataset[idx]
        
        with torch.no_grad():
            output = model(image.unsqueeze(0)) 
            _, predicted = torch.max(output, 1)
        
        actual_label_name = label_to_name[label]
        predicted_label_name = label_to_name[predicted.item()]
        
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 0.5) + 0.5 

        axs[i].imshow(image)
        axs[i].set_title(f"True: {actual_label_name}\nPred: {predicted_label_name}")
        axs[i].axis('off')  
    plt.show()

if show_confusion_matrix and evaluate:
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(range(len(cm))), yticklabels=list(range(len(cm))))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()