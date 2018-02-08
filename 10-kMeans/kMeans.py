# @Time    : 2018/2/7 下午12:33
# @Author  : zll
# @Site    : 
# @File    : kMeans.py
# @Software: PyCharm

import numpy as np


#####10.1 k-means聚类算法####################################################
##加载数据集
def loadDataSet(fileName):
    dataMat = []
    with open(fileName) as f1:
        content = f1.readlines()
        for line in content:
            line = list(map(float,line.strip("\n").split("\t")))
            dataMat.append(line)
    f1.close()
    return np.mat(dataMat)

##计算两个向量的欧氏距离
def distEclud(vecA,vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB,2)))

##为给定数据集构建k个随机质心
def randCent(dataSet,k):
    m,n = dataSet.shape
    centroids = np.mat(np.zeros((k,n)))

    for i in range(n):
        minValue = np.min(dataSet[:,i])
        maxValue = np.max(dataSet[:,i])
        Range = maxValue - minValue
        centroids[:,i] = minValue + Range * np.random.rand(k,1)
    return centroids

##k-means聚类算法
def kMeans(dataMat,k,distMeas=distEclud,createCent = randCent): #distMeas表示距离计算函数，createCent表示创建初始质心函数
    m,n = dataMat.shape
    clusteAssment = np.mat(np.zeros((m,2))) #存储每个点的簇分配结果:簇索引，误差
    centroids = createCent(dataMat,k) #初始质心
    clusterChanged = True #标志变量
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1

            for j in range(k): #寻找最近质心
                dist = distMeas(dataMat[i,:],centroids[j,:])
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            if clusteAssment[i,0] != minIndex:
                clusterChanged = True
            clusteAssment[i,:] = minIndex,minDist**2

        for c in range(k): #更新质心
            index = dataMat[np.nonzero(clusteAssment[:,0] == c)[0]]
            centroids[c,:] = np.mean(index,axis = 0)
        # print(centroids)
    return centroids,clusteAssment



#####10.3 二分k均值聚类算法####################################################
def biKmeans(dataMat,k,distMeas=distEclud):#distMeas表示距离计算函数
    m,n = dataMat.shape
    clusteAssment = np.mat(np.zeros((m,2))) #存储每个点的簇分配结果:簇索引，误差
    centroid0 = np.mean(dataMat,axis = 0).tolist()[0] #初始簇质心
    centList = [centroid0] #聚类中心
    for i in range(m):
        clusteAssment[i,1] = distEclud(dataMat[i,:],centroid0) ** 2

    while len(centList) < k:
        lowestSSE = np.inf

        for i in range(len(centList)): #尝试划分每一个簇
            ptsInCurrCluster = dataMat[np.nonzero(clusteAssment[:,0] == i)[0],:]#在第i簇的点
            centroids, splitClustAss = kMeans(ptsInCurrCluster,2) #分2簇
            sseSplit = np.sum(splitClustAss[:,1],axis = 0)
            sseNotSplit = np.sum(clusteAssment[np.nonzero(clusteAssment[:,0] != i)[0],1],axis = 0)
            print(sseSplit,sseNotSplit,sseSplit+sseNotSplit)

            if (sseSplit+sseNotSplit) < lowestSSE:
                bestCentToSplit = i #聚类误差最小的簇
                bestCentroid = centroids #2聚类的聚类中心
                bestClusterAss = splitClustAss.copy() #存储每个点的簇分配结果:簇索引，误差
                lowestSSE = sseSplit + sseNotSplit

        #更新簇分配结果
        bestClusterAss[np.nonzero(bestClusterAss[:,0] == 1)[0],0] = len(centList)
        bestClusterAss[np.nonzero(bestClusterAss[:,0] == 0)[0], 0] = bestCentToSplit
        centList[bestCentToSplit] = bestCentroid[0,:].tolist()[0]
        centList.append(bestCentroid[1,:].tolist()[0])
        clusteAssment[np.nonzero(clusteAssment[:,0] == bestCentToSplit)[0],:] = bestClusterAss
    return centList,clusteAssment


if __name__ == "__main__":

    #####10.1 k-means聚类算法####################################################
    # dataMat = loadDataSet("testSet.txt")
    # centroids = randCent(dataMat,2)
    # print("min:",min(dataMat[:,0]),min(dataMat[:,1]))
    # print("max:",max(dataMat[:,0]),max(dataMat[:,1]))
    # print("centroids:",centroids) #k个随机质心
    # print("distEclud:",distEclud(dataMat[0,:],dataMat[1,:]))
    # centroids,clusteAssment = kMeans(dataMat,4)

    #####10.3 二分k均值聚类算法####################################################
    dataMat = loadDataSet("testSet2.txt")
    centList, clusteAssment = biKmeans(dataMat,3)
    print(centList)

