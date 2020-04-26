import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
from torchvision import transforms


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


def Test():
    input_size = 784

    num_classes = 10
    num_epochs = 10
    batch_size = 50

    learning_rate = 0.001
    train_dataset = dsets.MNIST(root='./data/', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = dsets.MNIST(root='./data/', train=False, download=True, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                              )
    model = LogisticRegression(input_size, num_classes)
    lossfunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, x in enumerate(train_loader):
            images = x[0]
            labels = x[1]
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)

            optimizer.zero_grad()

            outputs = model(images)

            loss = lossfunction(outputs, labels)
            loss.backward()

            optimizer.step()

            if (epoch+1) % 1 == 0:
                print('Epoch[{}/{}], loss:{:.6f}'.format(epoch + 1, num_epochs, loss.item()))

