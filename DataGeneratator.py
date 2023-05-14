import torch
import random
import numpy as np
import matplotlib.pyplot as plt
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

xPosIndex = 0
yPosIndex = 1
timeIndex = 2
###传感器收集的时间
totalCollectTime = 10.0
posGap = 200
vGap = 5
maxCollectNum = 10
###速度变化
vChange = 5
###当收集的点数变化的时候,这里需要相应的变化
###协方差矩阵
covariance_matrix = np.eye(maxCollectNum)
mainRadarIndex = 0
minCollectEpoch = 0.1


def generate_normal_noise(mean, std, size):
    noise = np.random.normal(mean, std, size)
    return noise

class DataGenerator:
    def __init__(self, numRadar, numTrack ,diff, xv,yv, pos ,vChange,drawCurve=False):
        self.numRadar = numRadar
        self.numTrack = numTrack
        self.diff = diff
        self.xv = xv
        self.yv = yv
        self.pos = pos
        self.vChange = vChange
        self.time = np.arange(0.0, 1.0, 0.1)
        self.collect = int(totalCollectTime // minCollectEpoch)
        self.drawCurve = drawCurve
    def getRadarData(self):
        collectNums = torch.zeros(self.numRadar, 1, dtype=torch.int32)
        collectGap = np.random.uniform(1.0, 2.0, size=self.numRadar)
        collectGap = np.round(collectGap, decimals=1)
        ansTuple = []
        # 随机生成雷达观测误差
        diffVec = np.random.uniform(-self.diff, self.diff, self.numRadar)
        self.getRadarCollectNum(collectGap, collectNums)
        label = torch.zeros(self.numRadar, self.numTrack, dtype=torch.int64)
        xPos,yPos = self.getTrack()
        for radarIndex in range(0, self.numRadar):
            # 获得能够读取的点数
            curCanGettingPoints = collectNums[radarIndex]
            noise = generate_normal_noise(0, 1, curCanGettingPoints )
            ###x,y,t
            curAns = torch.zeros(self.numTrack, maxCollectNum, 3)
            for trackIndex in range(0, self.numTrack):
                label[radarIndex][trackIndex] = trackIndex
                nowTimeStamp = 0.0
                nowPointIndex = 0
                while nowPointIndex < curCanGettingPoints:
                    nowIndex = int(nowTimeStamp*10)
                    curAns[trackIndex][nowPointIndex][xPosIndex] = xPos[trackIndex][nowIndex] + diffVec[radarIndex] + noise[nowPointIndex]
                    curAns[trackIndex][nowPointIndex][yPosIndex] = yPos[trackIndex][nowIndex] + diffVec[radarIndex] + noise[nowPointIndex]
                    curAns[trackIndex][nowPointIndex][timeIndex] = nowTimeStamp
                    nowTimeStamp += collectGap[radarIndex]
                    nowPointIndex += 1
            ansTuple.append(curAns)
        # if(self.drawCurve):
        #     self.drawRadarDataCurve(ansTuple, collectNums, xPos, yPos)
        for index in range(0, len(ansTuple)):
            ansTuple[index] = torch.nn.functional.normalize(ansTuple[index], p=2.0, dim=1, eps=1e-12, out=None)
        return ansTuple, label, collectNums, xPos, yPos

    def getRadarCollectNum(self,collectGap, collectNums):
        for radarIndex in range(0, self.numRadar):
            collectNums[radarIndex][0] = int(totalCollectTime // collectGap[radarIndex])

    def getTrack(self):
        ansXPos = []
        ansYPos = []
        startPosX = generate_random_float_list(self.pos, self.pos + posGap, self.numTrack)
        startPosY = generate_random_float_list(self.pos, self.pos + posGap, self.numTrack)
        startVX = generate_random_float_list(self.xv, self.xv + vGap, self.numTrack)
        startVY = generate_random_float_list(self.yv, self.yv + vGap, self.numTrack)
        for trackIndex in range(0, self.numTrack):
            vX = 0.
            vY = 0.
            prexPos = 0.
            preyPos = 0.
            curPosVecX = []
            curPosVecY = []
            for point in range(0, self.collect):
                if point == 0:
                    vX = startVX[trackIndex]
                    vY = startVY[trackIndex]
                    prexPos = startPosX[trackIndex]
                    preyPos = startPosY[trackIndex]
                else:
                    if(point %10 == 0):
                        vX = generate_random_float_list(vX - self.vChange, vX + self.vChange, 1)[0]
                        vY = generate_random_float_list(vY - self.vChange, vY + self.vChange, 1)[0]
                    prexPos = (prexPos) + vX * (minCollectEpoch)
                    preyPos = (preyPos) + vY * (minCollectEpoch)
                curPosVecX.append(prexPos)
                curPosVecY.append(preyPos)
            ansXPos.append(curPosVecX)
            ansYPos.append(curPosVecY)
        return ansXPos, ansYPos

    def drawRadarDataCurve(self,radarData, collectNums,xPosG,yPosG):
        for radarIndex in range(0, self.numRadar):
            for trackIndex in range(0, self.numTrack):
                xPos = []
                yPos = []
                curCanGettingPoints = collectNums[radarIndex]
                for pointIndex in range(0, curCanGettingPoints):
                    xPos.append(radarData[radarIndex][trackIndex][pointIndex][xPosIndex].item())
                    yPos.append(radarData[radarIndex][trackIndex][pointIndex][yPosIndex].item())
                labelStr = 'CurveIndex' + str(radarIndex) + str(trackIndex)
                plt.scatter(xPos, yPos, label=labelStr, s=15)
        for track in range(0, self.numTrack):
            plt.plot(xPosG[track],yPosG[track],label=str(track))
        plt.title('Curves')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        return

    def drawRawTrack(self,xPos,yPos):
        plt.plot(xPos, yPos, label='true track')
        plt.show()














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