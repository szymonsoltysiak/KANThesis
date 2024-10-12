from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

label_to_find = 20

test_dataset = datasets.GTSRB(root='./data', split='test', transform=None, download=True)

label_index = next(i for i, (_, label) in enumerate(test_dataset) if label == label_to_find)

image, label = test_dataset[label_index]

image = np.array(image)

# Display the image
plt.imshow(image)
plt.title(f'Label: {label}')
plt.axis('off')
plt.show()