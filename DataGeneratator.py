import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

def generate_random_float_list(start, end, count):
    random_list = []
    for _ in range(count):
        random_list.append(random.uniform(start, end))
    return random_list




####dim0:传感器编号,dim1:捕获的轨迹数,dim2:channel4 1.传感器捕获x,2.传感器捕获的y,3.传感器捕获的vx,4.传感器捕获的vy,dim3:捕获的点数,
numRadar = 5
numTrack = 50
numGettingPoint = 5
xPosIndex = 0
yPosIndex = 1
vxIndex = 2
vyIndex = 3
### 传感器收集间隔
collectGap = 1
###速度变化
vChange = 10
###各个传感器在x和y方向上的采集误差
xDiff = generate_random_float_list(-5, 5, numRadar)
yDiff = generate_random_float_list(-7, 7, numRadar)
###当收集的点数变化的时候,这里需要相应的变化
###协方差矩阵
covariance_matrix = np.eye(numGettingPoint)

mainRadarIndex = 0



def generate_normal_noise(mean, std, size):
    noise = np.random.normal(mean, std, size)
    return noise

def drawRadarDataCurve (radarData):
    labelStr = ''
    for radarIndex in range(0, numRadar):
        for trackIndex in range(0, numTrack):
            xPos = []
            yPos = []
            labelStr = ''
            for pointIndex in range(0, numGettingPoint):
               xPos.append(radarData[radarIndex][trackIndex][xPosIndex][pointIndex].item())
               yPos.append(radarData[radarIndex][trackIndex][yPosIndex][pointIndex].item())
            labelStr = 'CurveIndex' + str(radarIndex) + str(trackIndex)
            plt.scatter(xPos, yPos, label=labelStr)
    plt.title('Curves')
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.legend()
    plt.show()
    return


####生成多个传感器的数据矩阵
####并且为各个数据打上标签

def getRadarData():
    ans = torch.zeros(numRadar, numTrack, 4, numGettingPoint)
    #随机生成雷达观测误差
    startPosX = generate_random_float_list(-2000, 2000, numTrack)
    startPosY = generate_random_float_list(-2000, 2000, numTrack)
    startVX = generate_random_float_list(-120, 120, numTrack)
    startVY = generate_random_float_list(-120, 120, numTrack)
    label = torch.zeros(numRadar, numTrack,dtype=torch.int64)

    for radarIndex in range(0, numRadar):
        for trackIndex in range(0, numTrack):
            preDataX = 0
            preDataY = 0
            label[radarIndex][trackIndex] = trackIndex
            for pointIndex in range(0, numGettingPoint):
                Vx = startVX[trackIndex]
                Vy = startVY[trackIndex]
                if pointIndex == 0:
                    ans[radarIndex][trackIndex][xPosIndex][pointIndex] = startPosX[trackIndex] + xDiff[radarIndex]
                    ans[radarIndex][trackIndex][xPosIndex][pointIndex] = generate_normal_noise(ans[radarIndex][trackIndex][xPosIndex][pointIndex], 1, 1)[0]
                    ans[radarIndex][trackIndex][yPosIndex][pointIndex] = startPosY[trackIndex] + yDiff[radarIndex]
                    ans[radarIndex][trackIndex][yPosIndex][pointIndex] = generate_normal_noise(ans[radarIndex][trackIndex][yPosIndex][pointIndex], 1, 1)[0]
                else:
                    Vx = random.uniform(Vx - vChange, Vx + vChange)
                    Vy = random.uniform(Vy - vChange, Vy + vChange)
                    ans[radarIndex][trackIndex][xPosIndex][pointIndex] = preDataX + Vx + xDiff[radarIndex]
                    ans[radarIndex][trackIndex][yPosIndex][pointIndex] = preDataY + Vy + yDiff[radarIndex]
                    ans[radarIndex][trackIndex][xPosIndex][pointIndex] = \
                    generate_normal_noise(ans[radarIndex][trackIndex][xPosIndex][pointIndex], 1, 1)[0]
                    ans[radarIndex][trackIndex][yPosIndex][pointIndex] = \
                    generate_normal_noise(ans[radarIndex][trackIndex][yPosIndex][pointIndex], 1, 1)[0]
                ans[radarIndex][trackIndex][vxIndex][pointIndex] = Vx
                ans[radarIndex][trackIndex][vyIndex][pointIndex] = Vy
                preDataX = ans[radarIndex][trackIndex][xPosIndex][pointIndex]
                preDataY = ans[radarIndex][trackIndex][yPosIndex][pointIndex]
    return ans, label

def radarDataToNet(radarInput):
    ansData = torch.zeros(numRadar, 4, numTrack, numTrack)
    for radarIndex in range(0, numRadar):
        for mainTrackIndex in range(0, numTrack):
            for subTrackIndex in range(0, numTrack):
                xPosMainRadar = radarInput[mainRadarIndex][mainTrackIndex][xPosIndex]
                yPosMainRadar = radarInput[mainRadarIndex][mainTrackIndex][yPosIndex]
                xVMainRadar = radarInput[mainRadarIndex][mainTrackIndex][vxIndex]
                yVMainRadar = radarInput[mainRadarIndex][mainTrackIndex][vyIndex]
                xPosSubRadar = radarInput[radarIndex][subTrackIndex][xPosIndex]
                yPosSubRadar = radarInput[radarIndex][subTrackIndex][yPosIndex]
                xVSubRadar = radarInput[radarIndex][subTrackIndex][vxIndex]
                yVSubRadar = radarInput[radarIndex][subTrackIndex][vyIndex]
                xPosDis = getMaDis(xPosMainRadar.numpy(),xPosSubRadar.numpy())
                yPosDis = getMaDis(yPosMainRadar.numpy(),yPosSubRadar.numpy())
                xVDis = getMaDis(xVMainRadar.numpy(),xVSubRadar.numpy())
                yVDis = getMaDis(yVMainRadar.numpy(),yVSubRadar.numpy())
                ansData[radarIndex][xPosIndex][mainTrackIndex][subTrackIndex] = xPosDis
                ansData[radarIndex][yPosIndex][mainTrackIndex][subTrackIndex] = yPosDis
                ansData[radarIndex][vxIndex][mainTrackIndex][subTrackIndex] = xVDis
                ansData[radarIndex][vyIndex][mainTrackIndex][subTrackIndex] = yVDis
    return ansData


def getMaDis(list1,list2):
    return mahalanobis(list1, list2, np.linalg.inv(covariance_matrix))


def one_hot(label, depth):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out