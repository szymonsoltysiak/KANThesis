import sys
sys.path.append("..")

from KAN import KAN
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import numpy as np

dataset = {}
train_input, train_label = make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=None)
test_input, test_label = make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=None)

dtype = torch.get_default_dtype()
dataset['train_input'] = torch.from_numpy(train_input).type(dtype)
dataset['test_input'] = torch.from_numpy(test_input).type(dtype)
dataset['train_label'] = torch.from_numpy(train_label[:,None]).type(dtype)
dataset['test_label'] = torch.from_numpy(test_label[:,None]).type(dtype)

X = dataset['train_input']
y = dataset['train_label']
plt.scatter(X[:,0], X[:,1], c=y[:,0])
plt.show()

model = KAN(width=[2,1], grid=3, k=3)

def train_acc():
    return torch.mean((torch.round(model(dataset['train_input'])[:,0]) == dataset['train_label'][:,0]).type(dtype))

def test_acc():
    return torch.mean((torch.round(model(dataset['test_input'])[:,0]) == dataset['test_label'][:,0]).type(dtype))

results = model.train(dataset, steps=50, metrics=(train_acc, test_acc))
print("train_acc: "+str(results['train_acc'][-1]))
print("test_acc: "+str(results['test_acc'][-1]))

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_torch = torch.from_numpy(grid_points).type(dtype)

with torch.no_grad():
    predictions = torch.round(model(grid_points_torch)[:, 0])

predictions = predictions.numpy().reshape(xx.shape)
plt.contourf(xx, yy, predictions, cmap=plt.cm.Spectral, alpha=0.8)

plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], edgecolor='k', marker='o')
plt.title("Model Predictions")
plt.show()

