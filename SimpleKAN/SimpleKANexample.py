from KAN import KAN
import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle

def create_dataset(f, 
                   n_var=2, 
                   ranges = [-1,1],
                   train_num=1000, 
                   test_num=1000,
                   normalize_input=False,
                   normalize_label=False,
                   device='cpu',
                   seed=0):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var,2)
    else:
        ranges = np.array(ranges)
        
    train_input = torch.zeros(train_num, n_var)
    test_input = torch.zeros(test_num, n_var)
    for i in range(n_var):
        train_input[:,i] = torch.rand(train_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
        test_input[:,i] = torch.rand(test_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
        
        
    train_label = f(train_input)
    test_label = f(test_input)
        
        
    def normalize(data, mean, std):
            return (data-mean)/std
            
    if normalize_input == True:
        mean_input = torch.mean(train_input, dim=0, keepdim=True)
        std_input = torch.std(train_input, dim=0, keepdim=True)
        train_input = normalize(train_input, mean_input, std_input)
        test_input = normalize(test_input, mean_input, std_input)
        
    if normalize_label == True:
        mean_label = torch.mean(train_label, dim=0, keepdim=True)
        std_label = torch.std(train_label, dim=0, keepdim=True)
        train_label = normalize(train_label, mean_label, std_label)
        test_label = normalize(test_label, mean_label, std_label)

    dataset = {}
    dataset['train_input'] = train_input.to(device)
    dataset['test_input'] = test_input.to(device)

    dataset['train_label'] = train_label.to(device)
    dataset['test_label'] = test_label.to(device)

    return dataset


f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]]**2)

model = KAN(width=[2, 1, 1], grid=3, k=3)
dataset = create_dataset(f, n_var=2)
model.train(dataset, steps=200)


with open('kan_model.pkl', 'wb') as file:
    pickle.dump(model, file)

x1_values = [0.0, 0.25, 0.5, 0.75, 1.0]  
num_x1 = len(x1_values)

x_values = torch.linspace(-1, 1, 100)

fig, axs = plt.subplots(1, num_x1, figsize=(15, 5))

for i, chosen_x1 in enumerate(x1_values):
    inputs = torch.stack([chosen_x1 * torch.ones_like(x_values), x_values], dim=1)

    function_outputs = f(inputs).detach().cpu().numpy()
    model_outputs = model(inputs).detach().cpu().numpy()

    axs[i].plot(x_values.numpy(), function_outputs, label='Actual Function Output', linewidth=2)
    axs[i].plot(x_values.numpy(), model_outputs, label='Model Output', linestyle='--', linewidth=2)
    axs[i].set_title(f'Outputs for x1 = {chosen_x1}')
    axs[i].set_xlabel('x2')
    axs[i].set_ylabel('Output')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()

