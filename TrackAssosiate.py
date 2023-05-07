import torch
import torch.nn as nn
import torch.optim as optim
import DataGeneratator as DG

numFeature = 30
###特征提取网络
class RNNClassifier(nn.GRU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(self.hidden_size*2*DG.maxCollectNum, numFeature)
        self.relu1 = nn.ReLU()
    def forward(self, input):
        result, hn = super().forward(input)
        result = result.view(self.hidden_size*2*DG.maxCollectNum)
        result = self.fc1(result)
        return result
###判别网络
class LogisticRegression(nn.Module):
    def __init__(self, input_size=2*numFeature):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.linear(x)
        out = self.softmax(out)
        return out


