import torch
import DataGeneratator as DG
import TrackAssosiate as TA
import torch.nn as nn
import torch.optim as optim
import testAcc
radarData, label = DG.getRadarData()
mainRadarLabel = label[DG.mainRadarIndex]

label.cuda()

print(label)
####绘制曲线
DG.drawRadarDataCurve(radarData)
feautureModel = TA.RNNClassifier(input_size=3, hidden_size=8, batch_first=True, num_layers=3, bidirectional=True)
classcifiyModel = TA.LogisticRegression()
###迁移到GPU上
feautureModel.cuda()
classcifiyModel.cuda()

optimizerF = optim.Adam(feautureModel.parameters(), lr=0.001)
optimizerC = optim.Adam(classcifiyModel.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
criterion.cuda()
lossPoint = []
def trainFunc(epochTime):
    for runEpoch in range(0, epochTime):
        running_loss = 0.0
        for currentRadar in range(0, DG.numRadar):
            if currentRadar == DG.mainRadarIndex:
                continue
            for mainRadarTrackIndex in range(0, DG.numTrack):
                for subRadarTrackIndex in range(0, DG.numTrack):
                    mainData = radarData[DG.mainRadarIndex][mainRadarTrackIndex].cuda()
                    subData = radarData[currentRadar][subRadarTrackIndex].cuda()
                    mainRadarOut = feautureModel.forward(mainData)
                    subRadarOut = feautureModel.forward(subData)
                    similar = torch.cat((mainRadarOut, subRadarOut), dim=0)
                    res = classcifiyModel.forward(similar)
                    res = res.cuda()
                    if label[DG.mainRadarIndex][mainRadarTrackIndex] == label[currentRadar][subRadarTrackIndex]:
                        loss = criterion(res, testAcc.matched)
                    else :
                        loss = criterion(res, testAcc.unmatched)
                    running_loss += loss.item()
                    optimizerC.zero_grad()
                    optimizerF.zero_grad()
                    loss.backward()
                    optimizerC.step()
                    optimizerF.step()
        if(runEpoch %100 == 0):
            print('epoch [%d] loss %f'%(runEpoch, running_loss))

        lossPoint.append(running_loss)
            # 每2000次迭代，输出loss的平均值
trainFunc(500)
DG.plotLoss(lossPoint)
torch.save(feautureModel, 'featureModel.pth')
torch.save(classcifiyModel, 'classModel.pth')

