import DataGeneratator as DG
import TrackAssosiate as TA
import torch
import random
matched = torch.ones(1)
unmatched = torch.zeros(1)
matched = matched.cuda()
unmatched = unmatched.cuda()


TP = 0.
FN = 0.
FP = 0.
TN = 0.
def testAcc() :
    state_dict = torch.load('feautureModel.pth')
    feautureModel = TA.RNNClassifier()
    if state_dict is not None:
        feautureModel.load_state_dict(state_dict['model'])
    feautureModel.eval()
    feautureModel.cuda()
    correct = 0.
    error = 0.
    global TP, FN, FP, TN
    for time in range (0,100):
        trainGenerator = DG.DataGenerator(numRadar=2, numTrack=10, diff=30, xv=30, yv=30, pos=2000, vChange=5, drawCurve=False)
        testData, testLabel, len, xPos, yPos = trainGenerator.getRadarData()
        testLabel.cuda()
        ansMatrix = torch.zeros(trainGenerator.numRadar,trainGenerator.numRadar,trainGenerator.numTrack,trainGenerator.numTrack)
        for firstRadar in range (0,trainGenerator.numRadar):
            for currentRadar in range(firstRadar + 1, trainGenerator.numRadar):
                for mainRadarTrackIndex in range(0,  trainGenerator.numTrack):
                    for subRadarTrackIndex in range(0,  trainGenerator.numTrack):
                        mainData = testData[firstRadar][mainRadarTrackIndex].cuda()
                        subData = testData[currentRadar][subRadarTrackIndex].cuda()
                        res = feautureModel.forward(mainData, len[firstRadar], subData, len[currentRadar])
                        ansMatrix[firstRadar][currentRadar][mainRadarTrackIndex][subRadarTrackIndex] = res
                        if testLabel[firstRadar][mainRadarTrackIndex] == testLabel[currentRadar][subRadarTrackIndex]:
                            if res >= 0.5:
                                TP = TP + 1
                            else:
                                FN = FN + 1
                        else:
                            if res >= 0.5:
                                FP = FP + 1
                            else:
                                TN = TN + 1
        resIndex = torch.argmax(ansMatrix,dim=3)
        for firstRadar in range (0,trainGenerator.numRadar):
            for currentRadar in range(firstRadar + 1, trainGenerator.numRadar):
                for mainRadarTrackIndex in range(0,  trainGenerator.numTrack):
                        if resIndex[firstRadar][currentRadar][mainRadarTrackIndex] == mainRadarTrackIndex:
                            correct = correct + 1
                        else:
                            error = error + 1
    print('acc is %f' %(correct / (correct + error)))
    ACC = (TP+ TN)/(TP + FP + TN + FN)
    RE = (TP)/(TP + FN)
    PR = (TP)/(TP + FP)
    F1 = 2/(1/PR + 1/RE)
    print('acc binary is %f , re is %f ,PR is %f,F1 is %f'% (ACC,RE,PR,F1))


