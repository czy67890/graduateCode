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


correct = 0.
error = 0.
def testAcc() :
    fModel = torch.load('featureModel.pth')

    fModel.eval()

    cModel = torch.load('classModel.pth')
    cModel.eval()
    trainGenerator = DG.DataGenerator(numRadar=30, numTrack=2, diff=5, xv=60, yv=60, pos=600, vChange=5)
    testData, testLabel, len, xPos, yPos = trainGenerator.getRadarData()
    trainGenerator.drawRadarDataCurve(testData, len, xPos, yPos)
    testLabel.cuda()
    cModel.eval()
    global correct, error
    for firstRadar in range (0,trainGenerator.numRadar):
        for currentRadar in range(0, trainGenerator.numRadar):
            for mainRadarTrackIndex in range(0,  trainGenerator.numTrack):
                for subRadarTrackIndex in range(0,  trainGenerator.numTrack):
                    mainData = testData[firstRadar][mainRadarTrackIndex].cuda()
                    subData = testData[currentRadar][subRadarTrackIndex].cuda()
                    mainRadarOut = fModel.forward(mainData, len[firstRadar])
                    subRadarOut = fModel.forward(subData, len[currentRadar])
                    similar = torch.cat((mainRadarOut, subRadarOut), dim=0)
                    res = cModel.forward(similar)
                    if testLabel[firstRadar][mainRadarTrackIndex] == testLabel[currentRadar][subRadarTrackIndex]:
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
