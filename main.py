import torch
import time
import DataGeneratator as DG
import TrackAssosiate as TA
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
radarData, label = DG.getRadarData()
testData, testLabel = DG.getRadarData()
mainRadarLabel = label[DG.mainRadarIndex]

label.cuda()
testLabel.cuda()

print(label)
####绘制曲线
DG.drawRadarDataCurve(radarData)
DG.drawRadarDataCurve(testData)
feautureModel = TA.RNNClassifier(input_size=3, hidden_size=8, batch_first=True, num_layers=3, bidirectional=True)
classcifiyModel = TA.LogisticRegression()

###迁移到GPU上
feautureModel.cuda()
classcifiyModel.cuda()

optimizerF = optim.Adam(feautureModel.parameters(), lr=0.001)
optimizerC = optim.Adam(classcifiyModel.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
criterion.cuda()

matched = torch.zeros(2)
unmatched = torch.zeros(2)

matched[0] = 1
unmatched[1] = 1


matched = matched.cuda()
unmatched = unmatched.cuda()

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
                        loss = criterion(res, matched)
                    else :
                        loss = criterion(res, unmatched)
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
trainFunc(5000)
DG.plotLoss(lossPoint)
correct = 0.
error = 0.
def testAcc() :
    global correct, error
    for currentRadar in range(0, DG.numRadar):
        if currentRadar == DG.mainRadarIndex:
            continue
        for mainRadarTrackIndex in range(0, DG.numTrack):
            for subRadarTrackIndex in range(0, DG.numTrack):
                mainData = testData[DG.mainRadarIndex][mainRadarTrackIndex].cuda()
                subData = testData[currentRadar][subRadarTrackIndex].cuda()
                mainRadarOut = feautureModel.forward(mainData)
                subRadarOut = feautureModel.forward(subData)
                similar = torch.cat((mainRadarOut, subRadarOut), dim=0)
                res = classcifiyModel.forward(similar)
                isMatched = torch.argmax(res,dim=0)
                if testLabel[DG.mainRadarIndex][mainRadarTrackIndex] == testLabel[currentRadar][subRadarTrackIndex]:
                    if isMatched == 0:
                        correct = correct + 1
                    else :
                        error = error + 1
                else:
                    if isMatched == 0:
                        error = error + 1
                    else:
                        correct = correct + 1


torch.save(feautureModel, 'featureModel.pth')
torch.save(classcifiyModel, 'classModel.pth')
testAcc()
print(correct)
print(error)
print(correct/(correct + error) )