import sys
import torch
import matplotlib.pyplot as plt

from CNNmodel import CNNModel
from CNNKanInSeries import CNNKan 
from KANConvModel.KANConvKANLinear import KANConvLinear

from datautils import transform, label_to_name
from PIL import Image

image_name = '20_2.jpg'
image_path = 'experimental_images/' + image_name

image = Image.open(image_path)
image = transform(image).unsqueeze(0)  


mode = 'CNN' # CNN or CNNKan or KANConvLinear
evaluate = False
show_examples = True

if mode == 'CNNKan':
    model = CNNKan()
elif mode == 'KANConvLinear':
    model = KANConvLinear()
elif mode == 'CNN':
    model = CNNModel()
else:
    raise ValueError('Invalid mode')
    sys.exit()

if mode == 'CNNKan':
    model.load_state_dict(torch.load('gtsrb_cnnkan_model.pth', weights_only=False))
elif mode == 'KANConvLinear':
    model.load_state_dict(torch.load('gtsrb_kanconvkanlinear_model.pth', weights_only=False))
if mode == 'CNN':  
    model.load_state_dict(torch.load('gtsrb_cnn_model.pth', weights_only=False))


with torch.no_grad():
    output = model(image)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    top3_prob, top3_catid = torch.topk(probabilities, 3)

for i in range(top3_prob.size(1)):
    label = label_to_name[top3_catid[0][i].item()]
    prob = top3_prob[0][i].item()
    print(f"Label: {top3_catid[0][i].item()} {label}, Probability: {prob:.4f}")


transformed_image = image.squeeze(0).permute(1, 2, 0) 
plt.imshow(transformed_image)
plt.title("Transformed Input Image")

plt.show()