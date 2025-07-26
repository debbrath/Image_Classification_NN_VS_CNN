# 🧠 Image Classification with NN vs CNN (Fashion MNIST) - PyTorch

This project compares the performance of a **fully connected Neural Network (NN)** and a **Convolutional Neural Network (CNN)** on the **Fashion MNIST** dataset using **PyTorch**.

---

## 🎯 Objective

- Train two models (NN and CNN) on Fashion MNIST
- Evaluate and compare their accuracy
- Visualize sample predictions
- Analyze why CNN performs better for image classification

---

## 📦 Dataset

- **Fashion MNIST** (from `torchvision.datasets`)
  - 60,000 training images
  - 10,000 test images
  - 28x28 grayscale images
  - 10 clothing categories (T-shirt/top, Trouser, Pullover, Dress, etc.)

---

## 🧠 Models

### 1️⃣ Neural Network (NN)
- Fully connected (Dense) layers only
- Layers: Flatten → Dense(128) → Dense(64) → Output(10)

### 2️⃣ Convolutional Neural Network (CNN)
- Layers: Conv2D → ReLU → MaxPool → Conv2D → MaxPool → Flatten → Dense → Output(10)

---

## 🛠️ Setup Instructions
### 1. Clone the Project

git clone https://github.com/your-username/image-classification-nn-vs-cnn.git
cd image-classification-nn-vs-cnn

#### 2. Create Virtual Environment (optional but recommended)
python -m venv venv
# Activate:
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
#### 3. Install Requirements
pip install -r requirements.txt

#### 4. Run the Project
python main.py


📈 Output Example
Test accuracy for NN and CNN

5 sample predictions per model

Comparison of accuracy:
Example:

objectivec
Copy
Edit
NN Accuracy:  0.8555  
CNN Accuracy: 0.9106  
CNN performed better than NN by 5.51%

🔍 Why CNN Performs Better
CNN captures spatial features using convolution filters.

CNN is translation invariant and parameter efficient for image data.

NN flattens image, losing spatial relationships.

✅ Requirements
Python ≥ 3.7 (✅ works with 3.13)

PyTorch

TorchVision

Matplotlib


