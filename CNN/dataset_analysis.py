import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import seaborn as sns
from collections import Counter
from datautils import transform

train_dataset = datasets.GTSRB(root='./data', split='train', transform=transform, download=True)
test_dataset = datasets.GTSRB(root='./data', split='test', transform=transform, download=True)

train_labels = [label for _, label in train_dataset]
test_labels = [label for _, label in test_dataset]

train_class_counts = Counter(train_labels)
test_class_counts = Counter(test_labels)

train_classes, train_counts = zip(*train_class_counts.items()) if train_class_counts else ([], [])
test_classes, test_counts = zip(*test_class_counts.items()) if test_class_counts else ([], [])

train_classes = list(train_classes)
train_counts = list(train_counts)
test_classes = list(test_classes)
test_counts = list(test_counts)

plt.figure(figsize=(12, 6))
sns.barplot(x=train_classes, y=train_counts, palette="viridis")
plt.title("Dystrybucja klas w zbiorze treningowym")
plt.xlabel("Numer klasy")
plt.ylabel("Ilość")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x=test_classes, y=test_counts, palette="viridis")
plt.title("Dystrybucja klas w zbiorze testowym")
plt.xlabel("Numer klasy")
plt.ylabel("Ilość")
plt.tight_layout()
plt.show()
