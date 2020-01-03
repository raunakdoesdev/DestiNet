import torch
import progressbar
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models.alexnet import AlexNet

device = torch.device('cuda')

# Define Datasets for Input
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define Network Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.cuda()

# Define Optimization Methods
import torch.optim as optim
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training Loop
for epoch in range(10):
    total = 0
    count = 0
    for inputs, labels in progressbar.progressbar(trainset):
        inputs = inputs.unsqueeze(0)
        labels = torch.cuda.LongTensor([labels])
        optimizer.zero_grad()
        outputs = net(inputs.cuda())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        count += 1
        total += loss.item()
    print(f"Epoch #{epoch} Training Loss = {total/count}")

    total = 0
    count = 0
    for inputs, labels in progressbar.progressbar(valset):
        inputs = inputs.unsqueeze(0)
        labels = torch.LongTensor([labels]).to(device)
        optimizer.zero_grad()
        outputs = net(inputs.to(device))
        loss = criterion(outputs, labels).to(device)

        count += 1
        total += loss.item()
    print(f"Epoch #{epoch} Validation Loss = {total/count}")
