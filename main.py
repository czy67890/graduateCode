import torch

import DataGeneratator as DG
import TrackAssosiate as TA
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
radarData, label = DG.getRadarData()
testData, testLabel = DG.getRadarData()
mainRadarLabel = label[DG.mainRadarIndex]


####绘制曲线
DG.drawRadarDataCurve(radarData)

feautureModel = TA.RNNClassifier(input_size=3, hidden_size=8, batch_first=True, num_layers=3, bidirectional=True)
classcifiyModel = TA.LogisticRegression()
optimizerF = optim.SGD(feautureModel.parameters(), lr=0.001, momentum=0.9)
optimizerC = optim.SGD(classcifiyModel.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
matched = torch.zeros(2)
unmatched = torch.zeros(2)
matched[0] = 1
unmatched[1] = 1
lossPoint = []
def trainFunc(epochTime):
    for runEpoch in range(0, epochTime):
        running_loss = 0.0
        for currentRadar in range(0, DG.numRadar):
            if currentRadar == DG.mainRadarIndex:
                continue
            for mainRadarTrackIndex in range(0, DG.numTrack):
                for subRadarTrackIndex in range(0, DG.numTrack):
                    mainRadarOut = feautureModel.forward(radarData[DG.mainRadarIndex][mainRadarTrackIndex])
                    subRadarOut = feautureModel.forward(radarData[currentRadar][subRadarTrackIndex])
                    similar = torch.cat((mainRadarOut, subRadarOut), dim=0)
                    res = classcifiyModel.forward(similar)
                    if label[DG.mainRadarIndex][mainRadarTrackIndex] == label[currentRadar][subRadarTrackIndex]:
                        loss = criterion(res, matched)
                    else :
                        loss = criterion(res, unmatched)
                    running_loss += loss.item()
                    optimizerC.zero_grad()
                    optimizerF.zero_grad()
                    loss.backward()
                    optimizerC.step()
                    optimizerF.step()
        #if(runEpoch %10000 == 0):
        lossPoint.append(running_loss)
            # 每2000次迭代，输出loss的平均值
trainFunc(20)
DG.plotLoss(lossPoint)
correct = 0.
error = 0.
def testAcc() :
    global correct , error
    for currentRadar in range(0, DG.numRadar):
        if currentRadar == DG.mainRadarIndex:
            continue
        for mainRadarTrackIndex in range(0, DG.numTrack):
            for subRadarTrackIndex in range(0, DG.numTrack):
                mainRadarOut = feautureModel.forward(radarData[DG.mainRadarIndex][mainRadarTrackIndex])
                subRadarOut = feautureModel.forward(radarData[currentRadar][subRadarTrackIndex])
                similar = torch.cat((mainRadarOut, subRadarOut), dim=0)
                res = classcifiyModel.forward(similar)
                isMatched = torch.argmax(res,dim=0)
                if label[DG.mainRadarIndex][mainRadarTrackIndex] == label[currentRadar][subRadarTrackIndex]:
                    if isMatched == 0:
                        correct = correct + 1
                    else :
                        error = error + 1
                else:
                    if isMatched == 0:
                        error = error + 1
                    else:
                        correct = correct + 1


testAcc()
print(correct)
print(error)
print(correct/(correct + error) )