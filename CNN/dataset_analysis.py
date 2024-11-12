import torch
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import seaborn as sns
from collections import Counter
from datautils import transform, label_to_name
from disturbances import get_average_color

# Load dataset
train_dataset = datasets.GTSRB(root='./data', split='train', transform=transform, download=True)
test_dataset = datasets.GTSRB(root='./data', split='test', transform=transform, download=True)

# Analyze and count occurrences of each class in training and test datasets
train_labels = [label for _, label in train_dataset]
test_labels = [label for _, label in test_dataset]

# Count occurrences of each class and sort by number of occurrences
train_class_counts = Counter(train_labels)
test_class_counts = Counter(test_labels)

# Sort class counts by number of occurrences
sorted_train_counts = sorted(train_class_counts.items(), key=lambda x: x[1], reverse=True)
sorted_test_counts = sorted(test_class_counts.items(), key=lambda x: x[1], reverse=True)

# Convert labels to names for better readability
sorted_train_labels = [label_to_name[label] for label, _ in sorted_train_counts]
sorted_train_values = [count for _, count in sorted_train_counts]

sorted_test_labels = [label_to_name[label] for label, _ in sorted_test_counts]
sorted_test_values = [count for _, count in sorted_test_counts]

# Plot histogram for the sorted class distribution in the training set
plt.figure(figsize=(12, 6))
sns.barplot(x=sorted_train_labels, y=sorted_train_values, palette="viridis")
plt.xticks(rotation=45, ha='right')  # Rotate labels to 45 degrees and align to the right
plt.title("Dystrybucja klas w zbiorze treningowym")
plt.xlabel("Klasa")
plt.ylabel("Ilość")
plt.tight_layout()
plt.show()

# Plot histogram for the sorted class distribution in the test set
plt.figure(figsize=(12, 6))
sns.barplot(x=sorted_test_labels, y=sorted_test_values, palette="viridis")
plt.xticks(rotation=45, ha='right')  # Rotate labels to 45 degrees and align to the right
plt.title("Dystrybucja klas w zbiorze testowym")
plt.xlabel("Klasa")
plt.ylabel("Ilość")
plt.tight_layout()
plt.show()

