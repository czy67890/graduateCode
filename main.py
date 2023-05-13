import torch
import DataGeneratator as DG
import TrackAssosiate as TA
import torch.nn as nn
import torch.optim as optim
import testAcc
import random

####绘制曲线
feautureModel = TA.RNNClassifier()
classcifiyModel = TA.LogisticRegression()
###迁移到GPU上
feautureModel.cuda()
classcifiyModel.cuda()

optimizerF = optim.Adam(feautureModel.parameters(), lr=0.001)
optimizerC = optim.Adam(classcifiyModel.parameters(), lr=0.001)
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

def trainFunc(epochTime,dataChange):
    global backWardTime
    global correctTrain
    global errorTrain
    for dataTime in range(0, dataChange):
        xv = random.randint(-30, 30)
        yv = random.randint(-20, 20)
        diff = random.randint(0, 0)
        pos = random.randint(-2000, 2000)
        trainGenerator = DG.DataGenerator(numRadar=30, numTrack=2, diff=diff, xv=xv, yv=yv, pos=pos, vChange=5)
        radarData, label, len, xPos,yPos = trainGenerator.getRadarData()
        trainGenerator.drawRadarDataCurve(radarData, len, xPos, yPos)
        for epo in range(0, epochTime):
            running_loss = 0.0
            for firstRadarIndex in range(0, trainGenerator.numRadar):
                for secondRadarIndex in range(firstRadarIndex + 1, trainGenerator.numRadar):
                    for mainRadarTrackIndex in range(0, trainGenerator.numTrack):
                        for subRadarTrackIndex in range(0, trainGenerator.numTrack):
                            mainData = radarData[firstRadarIndex][mainRadarTrackIndex].cuda()
                            subData = radarData[secondRadarIndex][subRadarTrackIndex].cuda()
                            mainRadarOut = feautureModel.forward(mainData, len[firstRadarIndex])
                            subRadarOut = feautureModel.forward(subData, len[secondRadarIndex])
                            similar = torch.cat((mainRadarOut, subRadarOut), dim=0)
                            res = classcifiyModel.forward(similar)
                            if label[firstRadarIndex][mainRadarTrackIndex] == label[secondRadarIndex][subRadarTrackIndex]:
                                if res >= 0.5:
                                    correctTrain = correctTrain + 1
                                else:
                                    errorTrain = errorTrain + 1
                                res = res.view(1, 1)
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
                            if backWardTime % 1000 == 0:
                                lossPoint.append(running_loss)
                                accPoint.append(correctTrain / (correctTrain + errorTrain))
                                print('backwardTime [%d] loss %f' % (backWardTime, running_loss))
                                print('current accuracy is %f' % (correctTrain / (correctTrain + errorTrain)))
                                correctTrain = 0.
                                errorTrain = 0.
                                running_loss = 0
            print('epoch [%d:%d]' % (dataTime, epo))
trainFunc(5, 3)
DG.plotLoss(lossPoint,accPoint)
torch.save(feautureModel, 'featureModel.pth')
torch.save(classcifiyModel, 'classModel.pth')
testAcc.testAcc()

