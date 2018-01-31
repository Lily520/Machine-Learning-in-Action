# @Time    : 2018/1/31 上午9:36
# @Author  : zll
# @Site    : 
# @File    : kNN.py.py
# @Software: PyCharm


import numpy as np
from collections import Counter
import os


###### 1. 创建数据集和标签 ##################################
def createDataSet():
    feature = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ["A","A","B","B"]
    return feature,labels

###### 2-1 k-近邻算法 ##################################
def classify0(inX,dataSet,labels,k):
    #inX:用于分类的输入向量  dataSet:训练集  labels:训练集对应的标签  k:最近邻居的数目

    diffMat = np.tile(inX,(dataSet.shape[0],1)) - dataSet #待分类的输入向量与每个训练数据做差
    distance = ((diffMat ** 2).sum(axis=1)) ** 0.5 #欧氏距离
    sortDistanceIndices = distance.argsort() #从小到大的顺序，返回对应索引值

    # classCount = {}  #统计k个邻居中，各个标签的个数
    votelabel = []
    for i in range(k):
        votelabel.append(labels[sortDistanceIndices[i]]) #投票所得第i个邻居的标签
    Xlabel = Counter(votelabel).most_common(1) #求votelabel中出现次数最多的元素和出现次数，该元素即为inX的标签
    return Xlabel[0][0]


###### 2-2 使用k-近邻算法改进约会网站的配对效果 ##################################
##将文件中的数据解析成训练样本矩阵和类标签向量
def file2matrix(filename):
    with open(filename) as f1:
        content = f1.readlines()
        lens = len(content)
        returnMat = np.zeros((lens,3))
        labels = []
        index = 0
        for line in content:
            line = line.strip("\n").split("\t")
            returnMat[index,:] = list(map(float,line[0:3]))
            labels.append(int(line[3]))
            index += 1
    f1.close()
    return returnMat,labels

##归一化特征
def autoNorm(dataSet):
    minValues = dataSet.min(axis = 0) #取array每一列的最小值
    maxValues = dataSet.max(axis = 0) #取array每一列的最大值
    ranges = maxValues - minValues

    normDataSet = np.zeros(dataSet.shape)
    normDataSet = dataSet - np.tile(minValues,(dataSet.shape[0],1))
    normDataSet = normDataSet/np.tile(ranges,(dataSet.shape[0],1))

    return normDataSet,ranges,minValues

##分类器针对约会网站的测试代码
def datingTest():
    testRatio = 0.1 #测试数据比例
    datingData, datingLabels = file2matrix("datingTestSet2.txt") #加载数据集
    normMat, ranges, minValues = autoNorm(datingData) #归一化
    m = normMat.shape[0] #数据个数
    numTest = int(m*0.1) #测试数据集个数

    #前numTest 条数据为测试数据，后面为训练数据
    errorCount = 0.00 #错误预测的样本个数
    for i in range(numTest):
        classifierResult = classify0(normMat[i,:],normMat[numTest:m,:],datingLabels[numTest:m],3)
        print("pred-label:",classifierResult,"----real-label:",datingLabels[i])
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("error rate:",errorCount/float(m))

##约会网站预测函数
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    inVec = [percentTats,ffMiles,iceCream] #待预测数据

    datingData, datingLabels = file2matrix("datingTestSet2.txt")  # 加载数据集
    normMat, ranges, minValues = autoNorm(datingData)  # 归一化
    norm_inVec = (inVec - minValues)/ranges #归一化的待预测数据

    classifierResult = classify0(norm_inVec,datingData,datingLabels,3) #分类结果
    print("you will probably like this person:",resultList[classifierResult])


###### 2-3 手写识别系统 ##########################################
##将图像格式化处理为一个向量
def img2vector(filename):
    with open(filename) as f1:
        returnVec = list(map(int,f1.read().replace("\n","")))
    return returnVec

##使用k近邻算法识别手写数字
def handwritingClasstest():
    #获取训练集
    trainingMat = [] #训练集特征
    trainLabel = [] #训练集标签
    for file in os.listdir("digits/trainingDigits"):
        label = int(file.split("_")[0])
        trainLabel.append(label)
        trainVec = img2vector("digits/trainingDigits/" + file)
        trainingMat.append(trainVec)
    trainingMat = np.array(trainingMat)

    #对测试集进行测试
    errorCount = 0.00
    num = 0
    for file in os.listdir("digits/testDigits"):
        num += 1
        real_label = int(file.split("_")[0]) #真实标签
        testVec = img2vector("digits/testDigits/" + file) #获取特征
        classifierResult = classify0(testVec, trainingMat, trainLabel, 3)  # 分类结果
        if classifierResult != real_label:
            errorCount += 1.00
        print("pred-label:", classifierResult, "----real-label:", real_label)

    print("the total error num:",errorCount)
    print("the total error rate:",errorCount/float(num))




if __name__ == "__main__":
    # ## 1. 创建数据集和标签
    # feature,labels = createDataSet()
    # # print(feature)
    # # print(labels)
    #
    #
    # ###### 2-1 k-近邻算法 ##################################
    # print(classify0([0,0],feature,labels,3))


    ###### 2-2 使用k-近邻算法改进约会网站的配对效果 ##################################
    # ##将文件中的数据解析成训练样本矩阵和类标签向量
    # datingData,datingLabels = file2matrix("datingTestSet2.txt")
    # # print(datingData)
    # # print(datingLabels)

    # ##归一化特征
    # normMat,ranges,minValues = autoNorm(datingData)
    # print(minValues)

    ##分类器针对约会网站的测试代码
    # datingTest()

    ##约会网站预测函数
    # classifyPerson()


    ###### 2-3 手写识别系统 ##########################################
    ##将图像格式化处理为一个向量
    # trainvec = img2vector("digits/trainingDigits/0_0.txt")
    # print(trainvec[0,32:63])

    ##使用k近邻算法识别手写数字
    handwritingClasstest()