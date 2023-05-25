import torch
import torch.nn as nn
import torch.optim as optim
import DataGeneratator as DG

Nhidden_size = 64
Ninput_size = 3
Nbatch_first = True
Nnum_layers = 1
###特征提取网络
class RNNClassifier(nn.GRU):
    def __init__(self):
        super().__init__(input_size=Ninput_size, hidden_size=Nhidden_size, batch_first=True, num_layers=Nnum_layers, bidirectional=True)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5,padding=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(3,3,kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01)  # 设置负斜率为0.01
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(2*3*Nhidden_size * Nnum_layers, 1),
            nn.Sigmoid()
        )

    def forward(self, input1,len1,input2,len2):
        input1 = input1.unsqueeze(0)
        input2 = input2.unsqueeze(0)
        package1 = nn.utils.rnn.pack_padded_sequence(input1, len1.cpu(), batch_first=self.batch_first,
                                                    enforce_sorted=False)
        package2 = nn.utils.rnn.pack_padded_sequence(input2, len2.cpu(), batch_first=self.batch_first,
                                                     enforce_sorted=False)
        output1, hidden1 = super().forward(package1)
        output2, hidden2 = super().forward(package2)
        # # output1 = output1.view(1, 10, 2*self.hidden_size)
        # # output2 = output2.view(1, 10, 2*self.hidden_size)
        hidden1 = hidden1.view(1,2*Nnum_layers,Nhidden_size)
        hidden2 = hidden2.view(1,2*Nnum_layers,Nhidden_size)
        # output1 = self.conv(output1)
        # output2 = self.conv(output2)
        hidden1 = self.conv(hidden1)
        hidden2 = self.conv(hidden2)
        dis = torch.abs(hidden1 - hidden2)
        x = dis.view(2*3*Nnum_layers*self.hidden_size)
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
