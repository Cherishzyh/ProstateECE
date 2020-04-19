import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, filters, stride=1):
        super(DoubleConv, self).__init__()
        self.conv1 = conv3x3(in_channels, filters, stride)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(filters, filters, stride)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class Unet(nn.Module):
    def __init__(self,  in_channels, out_channels):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.up5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.conv9 = nn.Conv2d(64, out_channels, 1)
        self.fc1 = nn.Linear(1*78400, 1*1000)
        self.fc2 = nn.Linear(1*1000, 1*2)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        up_5 = self.up5(c4)
        merge6 = torch.cat((up_5, c3), dim=1)
        c5 = self.conv6(merge6)
        up_6 = self.up6(c5)
        merge7 = torch.cat((up_6, c2), dim=1)
        c7 = self.conv7(merge7)
        up_7 = self.up7(c7)
        merge8 = torch.cat((up_7, c1), dim=1)
        c8 = self.conv8(merge8)
        c9 = self.conv9(c8)
        c9 = c9.view(-1, 1*280*280)
        out = self.fc2(self.fc1(c9))
        # out = nn.Softmax()(c9)
        return F.sigmoid(out)


def test():
    transform = transforms.Compose([transforms.Resize(40),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32),
                                    transforms.ToTensor()])

    train_dataset = dsets.CIFAR10(root='C:/PytorchLearning/data/',
                                  train=True,
                                  transform=transform,
                                  # transform=transforms.ToTensor(),
                                  download=True)

    test_dataset = dsets.CIFAR10(root='C:/PytorchLearning/data/',
                                 train=False,
                                 transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=100,
                                               shuffle=True,
                                               # num_workers=2
                                               )

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=100,
                                              shuffle=False,
                                              # num_workers=2
                                              )
    model = Unet(in_channels=3, out_channels=3)
    model = model.to(device)
    print(model)
    # nn.Sequential(*Unet)
    model.cuda()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(80):
        for i, data in enumerate(train_loader):
            # torch.size([100, 3, 32, 32])
            inputs, labels = data
            # torch.size([100])
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [%d / %d], Iter [%d / %d] Loss: %.4f' % (epoch + 1, 80, i + 1, 500, loss.data[0]))

        if (i + 1) % 20 == 0:
            lr /= 3
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    correct = 0
    total = 0

    for images, labels in test_loader:
        images = Variable(images.cuda())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

    correct += (predicted.cpu() == labels).sum()
    print('Accuracy of the model on the test image: %d %%' % (100 * correct / total))
    torch.save(model.state_dict(), 'resnet.pkl')


if __name__ == '__main__':

    test()