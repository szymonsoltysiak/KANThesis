import json
import matplotlib.pyplot as plt

models = ['CNN', 'CNNKKAN', 'CKAN']
metrics_data = {}
x_ticks = list(range(1, 11))

for model in models:
    metrics_file = f"{model.lower()}_metrics.json"
    try:
        with open(metrics_file, "r") as f:
            metrics_data[model] = json.load(f)
    except FileNotFoundError:
        print(f"Metrics file {metrics_file} not found. Skipping.")
        continue

colors = {
    'CNN': 'blue',
    'CNNKan': 'orange',
    'KANConvLinear': 'green'
}

labels = {
    'CNN': 'CNN',
    'CNNKan': 'CNNKAN',
    'KANConvLinear': 'CKAN'
}

max_tick = len(metrics_data['CNN']['training_losses'])
train_ticks = [1]+list(range(500, max_tick, 500))+[max_tick]

plt.figure(figsize=(10, 5))
for model, data in metrics_data.items():
    for i in range(len(data['training_losses'])):
        data['training_losses'][i] = data['training_losses'][i]/9
    plt.plot(range(1,len(data['training_losses'])+1), data['training_losses'], label=f"{labels[model]}", color=colors[model])
plt.xticks(train_ticks, fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(-20, max_tick+20)
plt.xlabel('Kroki', fontsize=14)
plt.ylabel('Funkcja straty', fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
for model, data in metrics_data.items():
    plt.plot(range(1,len(data['validation_losses'])+1), data['validation_losses'], label=f"{labels[model]}", color=colors[model])
plt.xticks(x_ticks, fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0.8, 10+0.2)
plt.xlabel('Epoki', fontsize=14)
plt.ylabel('Funkcja straty', fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
for model, data in metrics_data.items():
    plt.plot(range(1,len(data['validation_accuracies'])+1), data['validation_accuracies'], label=f"{labels[model]}", color=colors[model])
plt.xticks(x_ticks, fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0.8, 10+0.2)
plt.xlabel('Epoki', fontsize=14)
plt.ylabel('Dokładność (%)', fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
