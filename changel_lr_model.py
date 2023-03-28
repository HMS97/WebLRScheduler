
#create a simpe script to train the model on mnist dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import threading
from WebLRScheduler import WebLRScheduler

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.optim.lr_scheduler import LambdaLR

# Define transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Define data loaders
batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

model = Net().to(device)

# Define optimizer and loss function
current_learning_rate = 4e-4
optimizer = optim.SGD(model.parameters(), lr=current_learning_rate, momentum=0.5)
criterion = nn.CrossEntropyLoss()

# Train model
epochs = 100
# lr_lambda = lambda epoch: 0.8 ** epoch
scheduler = WebLRScheduler(optimizer,  total_iteration=len(train_loader)*epochs)



for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    print('Epoch: %d | Loss: %.3f' % (epoch + 1, running_loss / len(train_loader)) , 'lr', optimizer.param_groups[0]['lr'])

# Test model
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on test set: %d %%' % (100 * correct / total))
