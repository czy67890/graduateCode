import DataGeneratator as DG
import TrackAssosiate as TA
import torch

matched = torch.ones(1)
unmatched = torch.zeros(1)
unmatched = unmatched.view(1,1)
matched = matched.view(1,1)
print(matched)
print(unmatched)
matched = matched.cuda()
unmatched = unmatched.cuda()

testData, testLabel = DG.getRadarData()
testLabel.cuda()

DG.drawRadarDataCurve(testData)
correct = 0.
error = 0.
def testAcc() :
    fModel = torch.load('featureModel.pth')

    fModel.eval()

    cModel = torch.load('classModel.pth')

    cModel.eval()
    global correct, error
    for currentRadar in range(0, DG.numRadar):
        if currentRadar == DG.mainRadarIndex:
            continue
        for mainRadarTrackIndex in range(0, DG.numTrack):
            for subRadarTrackIndex in range(0, DG.numTrack):
                mainData = testData[DG.mainRadarIndex][mainRadarTrackIndex].cuda()
                subData = testData[currentRadar][subRadarTrackIndex].cuda()
                mainRadarOut = fModel.forward(mainData, DG.collectNums[DG.mainRadarIndex])
                subRadarOut = fModel.forward(subData, DG.collectNums[currentRadar])
                similar = torch.cat((mainRadarOut, subRadarOut), dim=0)
                res = cModel.forward(similar)
                if testLabel[DG.mainRadarIndex][mainRadarTrackIndex] == testLabel[currentRadar][subRadarTrackIndex]:
                    if res >= 0.5:
                        correct = correct + 1
                    else:
                        error = error + 1
                else:
                    if res >= 0.5:
                        error = error + 1
                    else:
                        correct = correct + 1
    print(correct)
    print(error)
    print(correct / (correct + error))
