import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from dataset import get_train_test_loaders


class cnnModel(nn.Module):
    #this class is where the models get built and trained on the training data set 
    def __init__(aslModel):
        super(cnnModel, aslModel).__init__()
        aslModel.conv1 = nn.Conv2d(1, 6, 3)
        aslModel.pool = nn.MaxPool2d(2, 2)
        aslModel.conv2 = nn.Conv2d(6, 6, 3)
        aslModel.pool = nn.MaxPool2d(2, 2)
        aslModel.conv3 = nn.Conv2d(6, 16, 3)
        aslModel.fc1 = nn.Linear(16 * 5 * 5, 120)
        aslModel.fc2 = nn.Linear(120, 48)
        aslModel.fc3 = nn.Linear(48, 24)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    net = cnnModel().float()
    crossEntroLoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    trainloader, _ = get_train_test_loaders()
    for epoch in range(10):  # loop over the dataset multiple times
        train(net, crossEntroLoss, optimizer, trainloader, epoch)
        scheduler.step()
    torch.save(net.state_dict(), "saveModel.pth")


def train(net, crossEntroLoss, optimizer, trainloader, epoch):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        input = Variable(data['image'].float())
        label = Variable(data['label'].long())
        optimizer.zero_grad()

        output = net(input)
        loss = crossEntroLoss(output, label[:, 0])
        loss.backward()
        optimizer.step()

        # print stats
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.6f' % (epoch, i, running_loss / (i + 1)))


if __name__ == '__main__':
    main()