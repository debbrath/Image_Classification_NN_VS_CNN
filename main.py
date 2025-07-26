import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 🔧 Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📦 Load Fashion MNIST
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

class_names = train_dataset.classes

# 🧠 NN Model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.fc(x)

# 🧠 CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
    
    # 📈 Training Function
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# 📊 Evaluation Function
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"✅ Test Accuracy: {accuracy:.4f}")
    return accuracy

# 🔍 Show Sample Predictions
def show_predictions(model, test_loader):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:5], labels[:5]
    outputs = model(images.to(device))
    _, preds = torch.max(outputs, 1)
    plt.figure(figsize=(10, 2))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"P: {class_names[preds[i]]}\nT: {class_names[labels[i]]}")
        plt.axis("off")
    plt.show()

# 🏁 Main
if __name__ == "__main__":
    print("\n🚀 Training NN...")
    nn_model = SimpleNN().to(device)
    nn_criterion = nn.CrossEntropyLoss()
    nn_optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
    train_model(nn_model, train_loader, nn_criterion, nn_optimizer)
    nn_acc = evaluate(nn_model, test_loader)
    show_predictions(nn_model, test_loader)

    print("\n🚀 Training CNN...")
    cnn_model = SimpleCNN().to(device)
    cnn_criterion = nn.CrossEntropyLoss()
    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
    train_model(cnn_model, train_loader, cnn_criterion, cnn_optimizer)
    cnn_acc = evaluate(cnn_model, test_loader)
    show_predictions(cnn_model, test_loader)

    # 🔍 Compare Results
    diff = (cnn_acc - nn_acc) * 100
    print(f"\n📊 Comparison:\nNN Accuracy:  {nn_acc:.4f}\nCNN Accuracy: {cnn_acc:.4f}")
    print(f"🎉 CNN performed better than NN by {diff:.2f}%")