import torch
import DataGeneratator as DG
import TrackAssosiate as TA
import torch.nn as nn
import torch.optim as optim
import testAcc
radarData, label = DG.getRadarData()
label.cuda()
####绘制曲线
DG.drawRadarDataCurve(radarData)
feautureModel = TA.RNNClassifier()
classcifiyModel = TA.LogisticRegression()
###迁移到GPU上
feautureModel.cuda()
classcifiyModel.cuda()

optimizerF = optim.Adam(feautureModel.parameters(), lr=0.0001)
optimizerC = optim.Adam(classcifiyModel.parameters(), lr=0.0001)
criterion = torch.nn.BCELoss(
    weight=None,
    size_average=None,
    reduction="mean",
)
criterion.cuda()
lossPoint = []
accPoint = []
backWardTime = 0
correctTrain = 0.
errorTrain = 0.

def trainFunc(epochTime):
    global backWardTime
    global correctTrain
    global errorTrain
    for runEpoch in range(0, epochTime):
        running_loss = 0.0
        for currentRadar in range(0, DG.numRadar):
            if currentRadar == DG.mainRadarIndex:
                continue
            for subRadarTrackIndex in range(0, DG.numTrack):
                for mainRadarTrackIndex in range(0, DG.numTrack):
                    mainData = radarData[DG.mainRadarIndex][mainRadarTrackIndex].cuda()
                    subData = radarData[currentRadar][subRadarTrackIndex].cuda()
                    mainRadarOut = feautureModel.forward(mainData, DG.collectNums[DG.mainRadarIndex])
                    subRadarOut = feautureModel.forward(subData, DG.collectNums[currentRadar])
                    similar = torch.cat((mainRadarOut, subRadarOut), dim=0)
                    res = classcifiyModel.forward(similar)
                    if label[DG.mainRadarIndex][mainRadarTrackIndex] == label[currentRadar][subRadarTrackIndex]:
                        if res >= 0.5:
                            correctTrain = correctTrain + 1
                        else:
                            errorTrain = errorTrain + 1
                        res = res.view(1,1)
                        res = res.cuda()
                        loss = criterion(res, testAcc.matched)
                    else:
                        if res >= 0.5:
                            errorTrain = errorTrain + 1
                        else:
                            correctTrain = correctTrain + 1
                        res = res.view(1, 1)
                        res = res.cuda()
                        loss = criterion(res, testAcc.unmatched)
                    backWardTime = backWardTime + 1
                    running_loss += loss.item()
                    optimizerC.zero_grad()
                    optimizerF.zero_grad()
                    loss.backward()
                    optimizerC.step()
                    optimizerF.step()
                    if backWardTime % 100 == 0:
                        lossPoint.append(running_loss)
                        accPoint.append(correctTrain / (correctTrain + errorTrain))
                        print('backwardTime [%d] loss %f' % (backWardTime, running_loss))
                        print('current accuracy is %f' % (correctTrain / (correctTrain + errorTrain)))
                        correctTrain = 0.
                        errorTrain = 0.
                        running_loss = 0

        print('epoch [%d]' % (runEpoch))
trainFunc(500)
DG.plotLoss(lossPoint,accPoint)

torch.save(feautureModel, 'featureModel.pth')
torch.save(classcifiyModel, 'classModel.pth')
testAcc.testAcc()

