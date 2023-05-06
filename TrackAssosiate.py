import torch
import torch.nn as nn
import torch.optim as optim
import DataGeneratator as DG

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.relu1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3,padding=1)
        self.relu2 = nn.Sigmoid()
        self.conv3 = nn.Conv2d(16, 4, kernel_size=3,padding=1)
        self.relu3 = nn.Sigmoid()
        self.fc1 = nn.Linear(4*DG.numTrack*DG.numTrack, DG.numTrack*DG.numTrack*3)
        self.fc2 = nn.Linear( DG.numTrack*DG.numTrack*3, DG.numTrack*DG.numTrack)
        self.softMax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(1, 4*DG.numTrack*DG.numTrack)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(DG.numTrack, DG.numTrack)
        x = self.softMax(x)
        return x


