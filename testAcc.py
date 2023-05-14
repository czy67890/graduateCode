import DataGeneratator as DG
import TrackAssosiate as TA
import torch

matched = torch.ones(1)
unmatched = torch.zeros(1)
matched = matched.cuda()
unmatched = unmatched.cuda()


correct = 0.
error = 0.
def testAcc() :
    state_dict = torch.load('feautureModel.pth')
    feautureModel = TA.RNNClassifier()
    if state_dict is not None:
        feautureModel.load_state_dict(state_dict['model'])
    feautureModel.eval()
    feautureModel.cuda()
    trainGenerator = DG.DataGenerator(numRadar=2, numTrack=30, diff=5, xv=60, yv=60, pos=600, vChange=5,drawCurve=True)
    testData, testLabel, len, xPos, yPos = trainGenerator.getRadarData()
    testLabel.cuda()
    global correct, error
    for firstRadar in range (0,trainGenerator.numRadar):
        for currentRadar in range(0, trainGenerator.numRadar):
            for mainRadarTrackIndex in range(0,  trainGenerator.numTrack):
                for subRadarTrackIndex in range(0,  trainGenerator.numTrack):
                    mainData = testData[firstRadar][mainRadarTrackIndex].cuda()
                    subData = testData[currentRadar][subRadarTrackIndex].cuda()
                    res = feautureModel.forward(mainData, len[firstRadar] ,subData, len[currentRadar])
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
