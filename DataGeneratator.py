import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

import DataGeneratator
####航迹组成---x,y,t的序列
####一个航迹就是一个n维3列的矩阵,其中n是时间段内获取的点数
####每个传感器捕获m个航迹
####拆解成航迹对并且观察是否关联即可
####选定一个主雷达
####让n个辅助雷达去匹配

def generate_random_float_list(start, end, count):
    random_list = []
    for _ in range(count):
        random_list.append(random.uniform(start, end))
    return random_list

####dim0:传感器编号,dim1:捕获的轨迹数,dim2:channel4 1.传感器捕获x,2.传感器捕获的y,3.传感器捕获的vx,4.传感器捕获的vy,dim3:捕获的点数,
numRadar = 3
numTrack = 30
xPosIndex = 0
yPosIndex = 1
timeIndex = 2
###传感器收集的时间
totalCollectTime = 10.0
### 传感器收集间隔
collectGap = generate_random_float_list(1.0, 2.0, numRadar)
maxCollectNum = 10
###速度变化
vChange = 5
###各个传感器在x和y方向上的采集误差
xDiff = generate_random_float_list(-20, 20, numRadar)
yDiff = generate_random_float_list(-20, 20, numRadar)
###当收集的点数变化的时候,这里需要相应的变化
###协方差矩阵
covariance_matrix = np.eye(maxCollectNum)
collectNums = torch.zeros(numRadar,dtype=torch.int32)
mainRadarIndex = 0
mainIndex = 0
subIndex = 1

def generate_normal_noise(mean, std, size):
    noise = np.random.normal(mean, std, size)
    return noise

def drawRadarDataCurve (radarData):
    for radarIndex in range(0, numRadar):
        for trackIndex in range(0, numTrack):
            xPos = []
            yPos = []
            curCanGettingPoints = collectNums[radarIndex]
            for pointIndex in range(0, curCanGettingPoints):
               xPos.append(radarData[radarIndex][trackIndex][0][pointIndex][xPosIndex].item())
               yPos.append(radarData[radarIndex][trackIndex][0][pointIndex][yPosIndex].item())
            labelStr = 'CurveIndex' + str(radarIndex) + str(trackIndex)
            plt.scatter(xPos, yPos, label=labelStr,s=15)
    plt.title('Curves')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    return

def getRadarCollectNum():
    for radarIndex in range(0,numRadar):
        collectNums[radarIndex] = int(totalCollectTime // collectGap[radarIndex])

####生成多个传感器的数据矩阵
####并且为各个数据打上标签
####最终生成的数据
####dim0 = 雷达编号,dim1 = 获取的点数 dim2 = 轨迹数 dim3 --x,y,t
def getRadarData():
    ansTuple = []
    #随机生成雷达观测误差
    getRadarCollectNum()
    startPosX = generate_random_float_list(-2000, 2000, numTrack)
    startPosY = generate_random_float_list(-2000, 2000, numTrack)
    startVX = generate_random_float_list(-30, 30, numTrack)
    startVY = generate_random_float_list(-30, 30, numTrack)
    label = torch.zeros(numRadar, numTrack, dtype=torch.int64)
    for radarIndex in range(0, numRadar):
        #获得能够读取的点数
        curCanGettingPoints = collectNums[radarIndex]
        ###x,y,t
        curAns = torch.zeros(numTrack, maxCollectNum, 3)
        Vx = 0
        Vy = 0
        for trackIndex in range(0, numTrack):
            label[radarIndex][trackIndex] = trackIndex
            preDataX = 0
            preDataY = 0
            nowTimeStamp = 0.0
            nowPointIndex = 0
            while nowPointIndex < curCanGettingPoints:
                if nowPointIndex == 0:
                    Vx = startVX[trackIndex]
                    Vy = startVY[trackIndex]
                    preDataX = startPosX[trackIndex]
                    preDataY = startPosY[trackIndex]
                Vx = random.uniform(Vx - vChange, Vx + vChange)
                Vy = random.uniform(Vy - vChange, Vy + vChange)
                preDataX = preDataX + Vx*(collectGap[radarIndex])
                preDataY = preDataY + Vy*(collectGap[radarIndex])
                curAns[trackIndex][nowPointIndex][xPosIndex] = \
                generate_normal_noise(preDataX, 1, 1)[0] + xDiff[radarIndex]
                curAns[trackIndex][nowPointIndex][yPosIndex] = \
                generate_normal_noise(preDataY, 1, 1)[0] + yDiff[radarIndex]
                curAns[trackIndex][nowPointIndex][timeIndex] = nowTimeStamp
                nowTimeStamp += collectGap[radarIndex]
                nowPointIndex += 1
        ###以下操作随机打散track
        indices = torch.randperm(curAns.size(0))
        for trackIndex in range(0, numTrack):
            label[radarIndex][trackIndex] = indices[trackIndex]
        curAns = curAns[indices]
        curAns = curAns.view(numTrack, 1, maxCollectNum, 3)
        ansTuple.append(curAns)
    return ansTuple, label


def getMaDis(list1,list2):
    return mahalanobis(list1, list2, np.linalg.inv(covariance_matrix))


def one_hot(label, depth):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out

def plotLoss(lossChange,accChange):
    x = range(0,len(lossChange))
    # 创建图形对象和两个轴对象
    fig, ax1 = plt.subplots()

    # 绘制第一个曲线
    ax1.plot(x, lossChange, 'r-')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color='r')
    ax1.tick_params('y', colors='r')

    # 创建第二个轴对象
    ax2 = ax1.twinx()
    # 绘制第二个曲线
    ax2.plot(x, accChange, 'g-')
    ax2.set_ylabel('accuracy', color='g')
    ax2.tick_params('y', colors='g')
    # 调整布局
    fig.tight_layout()
    # 显示图形
    plt.show()