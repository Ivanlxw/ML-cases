"""
Training MNIST dataset using PyTorch
"""

import torch
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from collections import OrderedDict

#Define transformation to normalize data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([.5],[.5])])

#Download and load training data
trainset = datasets.MNIST('~/pytorch/MNIST_data', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

#Building feed-forward network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # The layers
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()    


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

images, classes = next(iter(trainloader))
images = images.view(images.shape[0], -1)

epochs = 5
steps = 0
print_every = 400

for e in range(epochs):
    running_loss = 0
    for images, classes in iter(trainloader):
        steps += 1
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        optimizer.zero_grad()
    
        outputs = net(images)
        loss = criterion(outputs,classes)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        #print statistics
        if steps % print_every == 0:
            print("Training Loss: {:.3f}.. ".format(running_loss/print_every))
            running_loss = 0.0

total = 0
correct = 0
with torch.no_grad():
    for images, classes in iter(trainloader):
        images = images.view(images.shape[0],-1)
        output = net(images)
        _, predicted = torch.max(output.data, 1)
        total += classes.size(0)
        correct += (predicted == classes).sum().item()

print("Accuracy of network: %d %%" % (correct/total * 100)) 


