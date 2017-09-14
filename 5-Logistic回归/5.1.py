import numpy as np
import matplotlib.pyplot as plt

############1. 梯度上升法求回归系数#################################################################
#加载数据集
def loadDataSet(file):
    fr = open(file).readlines()
    feature = np.mat([list(map(float,line.strip("\n").split("\t")[0:-1])) for line in fr]) #(100,3)
    m,n = feature.shape
    x0 = np.ones((m,1))
    feature = np.hstack((x0,feature))
    label = np.mat([list(map(float,line.strip("\n").split("\t")[-1])) for line in fr]) #(100,1)
    return feature,label

#logistic函数
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#梯度上升法求回归系数
def gradAscent(feature,label):
    maxCycles = 500 #迭代次数
    m,n = feature.shape
    w = np.ones((n,1)) #初始化w
    alpha = 0.001
    for i in range(maxCycles):
        h = sigmoid(feature * w) #(m,1) 预测结果
        w = w + alpha * feature.T * (label - h) #梯度上升法更新w
    return w

##############2. 画出决策边界#########################################################################
def plotBestFit(feature,label,w):
    dataMat = np.hstack((feature[:,1:],label))
    data0 = dataMat[np.nonzero(dataMat[:,-1] == 0.00),:][0] #label=0
    data1 = dataMat[np.nonzero(dataMat[:, -1] == 1.00), :][0] #label = 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data0[:,0].tolist(),data0[:,1].tolist(),color = "r",marker = "o")
    ax.scatter(data1[:,0].tolist(),data1[:,1].tolist(),color = "g",marker = "x")
    x = np.arange(-4.0,4.0,0.1) #生成等差数列
    y = ((-w[0] - w[1] * x)/w[2]).tolist()[0]
    ax.plot(x,y)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

#########3. 随机梯度上升算法#############################################################
def stocGradAscent0(feature,label):
    m,n = feature.shape
    w = np.ones((n,1))
    alpha = 0.01
    for i in range(m):
        h = sigmoid(feature[i,:] * w)
        w = w + alpha * feature[i,:].T * (label[i,:] - h)
    return w

########4. 改进的随机梯度上升算法#######################################################
def stocGradAscent1(feature,label,numIter = 150):
    m,n = feature.shape
    w = np.ones((n,1))
    for i in range(numIter):
        dataIndex = list(range(m))
        for j in range(m):
            alpha = 4/(1.0 + i + j) + 0.01 #改进1：每次迭代都改变alpha
            randIndex = np.random.randint(0,len(dataIndex)) #随机生成dataIndex的某个索引 改进2：随机选取样本
            h = sigmoid(feature[dataIndex[randIndex],:] * w)
            w = w + alpha * feature[dataIndex[randIndex],:].T * (label[dataIndex[randIndex],:] - h)
            del dataIndex[randIndex] #删除list指定下标的元素
    return w

########5. 预测病马的死亡率###########################################
#加载数据集,与loadDataSet函数的区别是：这个函数没有在第一列之前增加全为1的列
def loadDataSet2(file):
    fr = open(file).readlines()
    feature = np.mat([list(map(float,line.strip("\n").split("\t")[0:-1])) for line in fr]) #(100,3)
    label = np.mat([float(line.strip("\n").split("\t")[-1]) for line in fr]).T #(100,1)
    return feature,label

def classifyVector(feature,w):
    h = sigmoid(feature * w)
    if h > 0.5 : return 1.00
    else: return 0.00

def colicTest(trainData,trainLabel,testData,testlabel):
    w = stocGradAscent1(trainData,trainLabel,500)
    error_sam = 0 #测试样本中预测错误的样本数
    m = testData.shape[0]
    for i in range(m):
        if classifyVector(testData[i,:],w) != testlabel[i,:]: error_sam += 1
    error_rate = float(error_sam)/float(testData.shape[0])
    print("error_rate: " ,error_rate)
    return error_rate


#求多次迭代的测试误差的平均值
def multiTest(trainFile,testFile):
    trainData, trainLabel = loadDataSet2(trainFile)
    testData, testlabel = loadDataSet2(testFile)
    numIter = 10
    error = 0.00

    for i in range(numIter):
        error += colicTest(trainData,trainLabel,testData,testlabel)
    error_rate = error/float(numIter)
    print("final error_rate: ",error_rate)


if __name__ == "__main__":
    ########1. 梯度上升法求回归系数#######################################################
    # feature,label = loadDataSet(".//machinelearninginaction//ch05//testSet.txt")
    # w = gradAscent(feature,label)
    # print(w)
    ########2. 画出决策边界##############################################################
    # plotBestFit(feature,label,w)
    #########3. 随机梯度上升算法############################################################
    # w0 = stocGradAscent0(feature,label)
    # print(w0)
    # plotBestFit(feature,label,w0)
    ########4. 改进的随机梯度上升算法#######################################################
    # w1 = stocGradAscent1(feature, label)
    # plotBestFit(feature, label, w1)
    # print(w1)
    ######5. 利用逻辑回归预测病马的死亡率######################################################
    trainFile = ".//machinelearninginaction//ch05//horseColicTraining.txt"
    testFile = ".//machinelearninginaction//ch05//horseColicTest.txt"
    multiTest(trainFile,testFile)