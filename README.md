# Traffic Sign Recognition Repository

## Models

### 1. Classic CNN

**File:** `CNNModel.py`  
A classical CNN with convolutional layers followed by a multi-layer perceptron (MLP).  
**Based on:** [Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)

### 2. CNN KAN

**File:** `CNNKanInSeries.py`  
A model with convolutional layers followed by a Kolmogorov-Arnold Network (KAN).  
**Based on:** [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)

### 3. Mixed Model

**Files:** `KANConcModel.py` / `KANConvKANLinear.py`  
A hybrid architecture combining classical convolutional layers with KAN-convolutional layers, followed by a KAN.  
**Based on:** [Convolutional Kolmogorov-Arnold Networks](https://arxiv.org/abs/2406.13155)

## Dataset

**GTSRB - German Traffic Sign Recognition Benchmark**  
Dataset link: [GTSRB on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

## Kolmogorov-Arnold Network Implementations

### Simple KAN

Original implementation of the Kolmogorov-Arnold Network.  
**Source:** [pykan GitHub Repository](https://github.com/KindXiaoming/pykan)  
**Paper:** [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)

### Fast KAN

An optimized version of KAN for faster training and inference.  
**Source:** [Fast-KAN GitHub Repository](https://github.com/ZiyaoLi/fast-kan)
