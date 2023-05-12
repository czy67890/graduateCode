import torch
import DataGeneratator as DG
import TrackAssosiate as TA
import torch.nn as nn
import torch.optim as optim
import testAcc
radarData, label = DG.getRadarData()
label.cuda()
print(label)

print(torch.__version__)
####绘制曲线
DG.drawRadarDataCurve(radarData)
feautureModel = TA.RNNClassifier(input_size=3, hidden_size=8, batch_first=True, num_layers=3, bidirectional=True)
classcifiyModel = TA.LogisticRegression()
###迁移到GPU上
feautureModel.cuda()
classcifiyModel.cuda()

optimizerF = optim.Adam(feautureModel.parameters(), lr=0.0001)
optimizerC = optim.Adam(classcifiyModel.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
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
            for mainRadarTrackIndex in range(0, DG.numTrack):
                for subRadarTrackIndex in range(0, DG.numTrack):
                    backWardTime = backWardTime + 1
                    mainData = radarData[DG.mainRadarIndex][mainRadarTrackIndex].cuda()
                    subData = radarData[currentRadar][subRadarTrackIndex].cuda()
                    mainRadarOut = feautureModel.forward(mainData)
                    subRadarOut = feautureModel.forward(subData)
                    similar = torch.cat((mainRadarOut, subRadarOut), dim=0)
                    res = classcifiyModel.forward(similar)
                    res = res.cuda()
                    isMatched = torch.argmax(res, dim=0)
                    if label[DG.mainRadarIndex][mainRadarTrackIndex] == label[currentRadar][subRadarTrackIndex]:
                        loss = criterion(res, testAcc.matched)
                        if isMatched == 0:
                            correctTrain = correctTrain + 1
                        else:
                            errorTrain = errorTrain + 1
                    else:
                        loss = criterion(res, testAcc.unmatched)
                        if isMatched == 0:
                            errorTrain = errorTrain + 1
                        else:
                            correctTrain = correctTrain + 1
                    running_loss += loss.item()
                    optimizerC.zero_grad()
                    optimizerF.zero_grad()
                    loss.backward()
                    optimizerC.step()
                    optimizerF.step()
                    if backWardTime % 100 == 0:
                        lossPoint.append(running_loss)
                        accPoint.append(correctTrain/(correctTrain + errorTrain))
                        print('backwardTime [%d] loss %f' % (backWardTime, running_loss))
                        print('current accuracy is %f'%(correctTrain/(correctTrain + errorTrain)))
                        correctTrain = 0.
                        errorTrain = 0.
                        running_loss = 0
        print('epoch [%d]' % (runEpoch))
trainFunc(500)
DG.plotLoss(lossPoint,accPoint)

torch.save(feautureModel, 'featureModel.pth')
torch.save(classcifiyModel, 'classModel.pth')
testAcc.testAcc()

