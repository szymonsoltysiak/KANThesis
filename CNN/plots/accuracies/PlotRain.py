import json
import matplotlib.pyplot as plt

with open('acc_rain.json', 'r') as f:
    data = json.load(f)

coeff = data['coeff']
models = ['CNN', 'CNNKAN', 'CKAN', 'CNN_rain', 'CNNKAN_rain', 'CNN_mix', 'CNNKAN_mix']
accuracies = {model: data[model] for model in models}

plt.figure(figsize=(10, 6))
for model, acc in accuracies.items():
    plt.plot(coeff, acc, label=model, marker='o')

plt.xlabel("Współczynnika pokrycia", fontsize=12)
plt.ylabel("Dokładność (%)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend( fontsize=10)

plt.tight_layout()
plt.show()
