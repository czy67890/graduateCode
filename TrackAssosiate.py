import torch
import torch.nn as nn
import torch.optim as optim
import DataGeneratator as DG

numFeature = 30
Nhidden_size = 64
Ninput_size = 3
Nbatch_first = True
Nnum_layers = 2
###特征提取网络
class RNNClassifier(nn.GRU):
    def __init__(self):
        super().__init__(input_size=Ninput_size, hidden_size=Nhidden_size, batch_first=True, num_layers=Nnum_layers, bidirectional=False)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.ReLU()
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(Nhidden_size * 10, 1),
            # nn.ReLU(),
            # nn.Linear(32, 1),
            nn.Sigmoid()
        )



    def forward(self, input1,len1,input2,len2):
        input1 = input1.unsqueeze(0)
        input2 = input2.unsqueeze(0)
        output1, hidden1 = super().forward(input1)
        output2, hidden2 = super().forward(input2)

        dis = torch.abs(output1 - output2)
        x = dis.view(1*10* self.hidden_size)
        x = self.fc(x)
        return x
###判别网络
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()

        # 卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.ReLU()
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(3*2*Nhidden_size*Nnum_layers, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(1,2*Nnum_layers,Nhidden_size)
        x = self.conv(x)
        x = x.view(3*2*Nhidden_size*Nnum_layers)
        x = self.fc(x)
        return x
