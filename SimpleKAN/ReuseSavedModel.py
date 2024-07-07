from KAN import KAN
import matplotlib.pyplot as plt
import torch
import pickle

with open('kan_model.pkl', 'rb') as file:
    model = pickle.load(file)

f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]]**2)

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

