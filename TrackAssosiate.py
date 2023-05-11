import torch
import torch.nn as nn
import torch.optim as optim
import DataGeneratator as DG

numFeature = 30
attention_size = 9
###特征提取网络
class RNNClassifier(nn.GRU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = nn.Linear(self.hidden_size * 2, attention_size)  # *2 for bidirectional
        self.softmax = nn.Softmax(dim=1)
        self.fc1 = nn.Linear(self.hidden_size * 2, attention_size)
    def forward(self, input):
        output, hidden = super().forward(input)
        # Attention
        # Concatenate the hidden states of both directions
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # Attention
        attention_scores = self.attention(hidden_cat)
        attention_weights = self.softmax(attention_scores)
        hidden_cat = self.fc1(hidden_cat)
        hidden_cat = hidden_cat * attention_weights
        return hidden_cat
###判别网络
class LogisticRegression(nn.Module):
    def __init__(self, input_size=2*attention_size):
        super(LogisticRegression, self).__init__()
        self.cov1 = nn.Conv2d(2,3,kernel_size=1)
        self.linear = nn.Linear(3*attention_size, 2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view(2,3,3)
        x = self.cov1(x)
        x = x.view(3*attention_size)
        out = self.linear(x)
        out = self.softmax(out)
        return out


