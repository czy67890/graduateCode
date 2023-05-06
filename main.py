import torch

import DataGeneratator as DG
import TrackAssosiate as TA
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
radarData, label = DG.getRadarData()

mainRadarLabel = label[DG.mainRadarIndex]
print(mainRadarLabel)
# DG.drawRadarDataCurve(radarData)

netInput = DG.radarDataToNet(radarData)
print(netInput[0].size())
netInput = netInput.to(device)
###outPut


#创建模型实例



model = TA.CNN()
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
def tarinFunc(epoch):
    for runEpoch in range(0, epoch):
        running_loss = 0.0
        for currentRadar in range(0, DG.numRadar):
            out = model.forward(netInput[currentRadar])
            realLabel = label[currentRadar]
            realLabel = DG.one_hot(realLabel,DG.numTrack)
            realLabel = realLabel.to(device)
            realLabel.requires_grad_()
            loss = 0
            for trackIndex in range(0,DG.numTrack):
                loss = loss + criterion(out[trackIndex], realLabel[trackIndex])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('loss is %3f', loss.item())
        # 每2000次迭代，输出loss的平均值

tarinFunc(20)


def TestAcc():
    correct = 0
    error = 0
    for currentRadar in range(0, DG.numRadar):
        out = model.forward(netInput[currentRadar])
        realLabel = label[currentRadar]
        realLabel = DG.one_hot(realLabel, DG.numTrack)

        for trackIndex in range(0,DG.numTrack):
            if realLabel[trackIndex].argmax() == out[trackIndex].argmax():
                correct = correct + 1
            else :
                error = error + 1
    print(correct)
    print(error)
    print(correct/(correct+error))

TestAcc()