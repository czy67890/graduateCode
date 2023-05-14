import torch
import DataGeneratator as DG
import TrackAssosiate as TA
import torch.nn as nn
import torch.optim as optim
import testAcc
import random
import time
####绘制曲线

##迁移到GPU上

# state_dict = None
# feautureModel = TA.RNNClassifier()
# if state_dict is not None:
#     feautureModel.load_state_dict(state_dict['model'])
# feautureModel.cuda()
#
# optimizerF = optim.Adam(feautureModel.parameters(), lr=0.01)
# criterion = torch.nn.BCELoss()
# criterion.cuda()
# lossPoint = []
# accPoint = []
# backWardTime = 0
# correctTrain = 0.
# errorTrain = 0.
# def trainFunc(epochTime, dataChange):
#     global backWardTime
#     global correctTrain
#     global errorTrain
#     for dataTime in range(0, dataChange):
#         xv = random.randint(-30, 30)
#         yv = random.randint(-20, 20)
#         diff = random.randint(0, 10)
#         pos = random.randint(-2000, 2000)
#         trainGenerator = DG.DataGenerator(numRadar=5, numTrack=2, diff=diff, xv=xv, yv=yv, pos=pos, vChange=5)
#         radarData, label, len, xPos,yPos = trainGenerator.getRadarData()
#         jump = False
#         epo = 0
#         while epo < epochTime:
#             epo = epo + 1
#             if jump:
#                 break
#             running_loss = 0.0
#             for firstRadarIndex in range(0, trainGenerator.numRadar):
#                 if jump:
#                     break
#                 for secondRadarIndex in range(firstRadarIndex + 1, trainGenerator.numRadar):
#                     if jump:
#                         break
#                     for mainRadarTrackIndex in range(0, trainGenerator.numTrack):
#                         if jump:
#                             break
#                         for subRadarTrackIndex in range(0, trainGenerator.numTrack):
#                             mainData = radarData[firstRadarIndex][mainRadarTrackIndex].cuda()
#                             subData = radarData[secondRadarIndex][subRadarTrackIndex].cuda()
#                             res = feautureModel.forward(mainData, len[firstRadarIndex],subData, len[secondRadarIndex])
#                             if label[firstRadarIndex][mainRadarTrackIndex] == label[secondRadarIndex][subRadarTrackIndex]:
#                                 if res >= 0.5:
#                                     correctTrain = correctTrain + 1
#                                 else:
#                                     errorTrain = errorTrain + 1
#                                 loss = criterion(res, testAcc.matched)
#                             else:
#                                 if res >= 0.5:
#                                     errorTrain = errorTrain + 1
#                                 else:
#                                     correctTrain = correctTrain + 1
#                                 loss = criterion(res, testAcc.unmatched)
#                             backWardTime = backWardTime + 1
#                             running_loss += loss.item()
#                             optimizerF.zero_grad()
#                             loss.backward()
#                             optimizerF.step()
#                             if backWardTime % 100 == 0:
#                                 lossPoint.append(running_loss)
#                                 acc = (correctTrain) / (correctTrain + errorTrain)
#                                 accPoint.append(acc)
#                                 print('backwardTime [%d] loss %f' % (backWardTime, running_loss))
#                                 print('current accuracy is %f' % acc)
#                                 correctTrain = 0.
#                                 errorTrain = 0.
#                                 if(acc > 0.999 and running_loss <= 0.0002):
#                                    jump = True
#                                    break
#                                 running_loss = 0
#             torch.save({'model': feautureModel.state_dict()}, 'feautureModel.pth')
#             print('epoch [%d:%d]' % (dataTime, epo))
# trainFunc(500, 30)
# torch.save({'model': feautureModel.state_dict()}, 'feautureModel.pth')
# DG.plotLoss(lossPoint, accPoint)
testAcc.testAcc()

