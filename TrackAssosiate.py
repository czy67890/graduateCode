import torch
import torch.nn as nn
import torch.optim as optim
import DataGeneratator as DG

numFeature = 30
Nhidden_size = 8

Ninput_size = 3
Nbatch_first = True
Nnum_layers = 2
###特征提取网络
class RNNClassifier(nn.GRU):
    def __init__(self):
        super().__init__(input_size=Ninput_size, hidden_size=Nhidden_size, batch_first=True, num_layers=Nnum_layers, bidirectional=True)
    def forward(self, input,len):
        package = nn.utils.rnn.pack_padded_sequence(input, len.cpu(), batch_first=Nbatch_first,
                                                    enforce_sorted=False)
        output, hidden = super().forward(input)
        hidden = hidden.view(2*self.hidden_size*Nnum_layers)
        return hidden
###判别网络
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()

        # 卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.ReLU()
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(2*Nhidden_size*Nnum_layers, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 展平特征图
        x = x.view(2,2*Nnum_layers,Nhidden_size)
        x = self.conv(x)
        x = x.view(2*Nhidden_size*Nnum_layers)
        x = self.fc(x)
        return x
