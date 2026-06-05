
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
torch.manual_seed(42)
np.random.seed(42)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(64)
        self.conv9 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(64)
        self.conv10 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(64)
        self.conv11 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64)
        self.conv13 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(64)
        self.conv14 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn14 = nn.BatchNorm2d(64)
        self.conv15 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn15 = nn.BatchNorm2d(64)
        self.conv16 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn16 = nn.BatchNorm2d(64)
        self.conv17 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn17 = nn.BatchNorm2d(64)
        self.conv18 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn18 = nn.BatchNorm2d(64)
        self.conv19 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn19 = nn.BatchNorm2d(64)
        self.conv20 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn20 = nn.BatchNorm2d(64)
        self.conv21 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(64)
        self.conv22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(64)
        self.conv23 = nn.Conv2d(64, 10, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.relu(self.bn13(self.conv13(x)))
        x = F.relu(self.bn14(self.conv14(x)))
        x = F.relu(self.bn15(self.conv15(x)))
        x = F.relu(self.bn16(self.conv16(x)))
        x = F.relu(self.bn17(self.conv17(x)))
        x = F.relu(self.bn18(self.conv18(x)))
        x = F.relu(self.bn19(self.conv19(x)))
        x = F.relu(self.bn20(self.conv20(x)))
        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        x = self.conv23(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return F.softmax(x, dim=1)

# ── Data loading ──────────────────────────────────────────────────────────────
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform)
# Fast mode — use subset for quicker training on demo
if __FAST_MODE__:
    train_dataset = torch.utils.data.Subset(train_dataset, range(10000))
    test_dataset  = torch.utils.data.Subset(test_dataset,  range(2000))
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=64, shuffle=False)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = UNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.99, nesterov=True, weight_decay=0.01)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}")

model.eval()
correct = 0
total   = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"Final Accuracy: {accuracy:.2f}%")
