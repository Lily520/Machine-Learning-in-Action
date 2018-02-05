# @Time    : 2018/1/31 下午4:07
# @Author  : zll
# @Site    : 
# @File    : adaboost.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt

####7.3 基于单层决策树构建弱分类器##################################################
##构建一个简单数据集
def loadSimpData():
    dataMat = np.matrix([[1.0,2.1],
                         [2.0,1.1],
                         [1.3,1.0],
                         [1.0,1.0],
                         [2.0,1.0]])
    labels = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,labels

##单层决策树生成函数
def stumpClassify(dataMatrix,dimen,thresh,threshIneq): #通过阈值比较对数据进行分类
    #dimen表示数据的第dimen个维度，thresh表示阈值，threshIneq表示不等号

    returnLabel = np.ones((dataMatrix.shape[0],1)) #预测数据的标签
    if threshIneq == "lt": #小于
        returnLabel[dataMatrix[:,dimen] <= thresh] = -1.0
    else: #大于
        returnLabel[dataMatrix[:, dimen] > thresh] = -1.0
    return returnLabel

def buildStump(dataArr,Labels,D): #遍历stumpClassify的所有可能输入值，找到最佳单层决策树
    #D表示数据的权重向量

    dataMatrix = np.mat(dataArr)
    Labels = np.mat(Labels).T
    m,n = dataMatrix.shape
    numSteps = 10.0 #用于在特征的所有可能值上进行遍历
    minError = np.inf #最小错误率设为无穷大
    bestStump = {} #记录最优决策树的相关信息
    bestClassPre = np.zeros((m,1)) #记录最佳决策树的预测标签

    for i in range(n): #对每一维特征
        minVal = dataMatrix[:,i].min()
        maxVal = dataMatrix[:,i].max()
        stepSize = (maxVal - minVal)/numSteps #步长
        for j in range(-1,int(numSteps)+1): #在第i维的所有可能值上进行遍历
            thresh = minVal + float(j) * stepSize #阈值
            for ineqal in ["lt","gt"]:#不等号
                predictLabels = stumpClassify(dataMatrix,i,thresh,ineqal) #预测标签
                errArr = np.ones((m,1)) #误差矩阵
                errArr[predictLabels == Labels] = 0
                weightedError = D.T * errArr
                # print("dim ",i,", thresh ",thresh, ",ineqal ",ineqal,",weightedError ",weightedError)

                if weightedError < minError:
                    minError = weightedError
                    bestStump['dim'] = i
                    bestStump['thresh'] = thresh
                    bestStump['ineqal'] = ineqal
                    bestClassPre = predictLabels.copy()
    return bestStump,minError,bestClassPre


####7.4 完整adaboost算法实现##################################################
##基于单层决策树的Adaboost训练过程
def adaBoostTrainDS(dataArr,Labels,numIters = 40):#dataArr,Labels,numIters分别表示数据集，类别标签和迭代次数
    weakClassArr = [] #存储各个单层决策树的相关信息
    m,n = np.shape(dataArr)
    D = np.mat(np.ones((m,1))/m) #数据权重
    aggClassEst = np.zeros((m,1)) #每个数据的类别估计累计值

    for i in range(numIters):
        bestStump, minError, bestClassPre = buildStump(dataArr,Labels,D)
        alpha = float(0.5 * np.log((1-minError)/max(minError,1e-16))) #每个弱分类器的权重，max(minError,1e-16)防止除零溢出
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)

        expon = np.multiply(-1 * alpha * np.mat(Labels).T, bestClassPre)
        # print("D:", D)
        D = np.multiply(D,np.exp(expon))
        D = D/D.sum() #更新数据权重D
        aggClassEst += alpha * bestClassPre
        # print("bestClassPre:",bestClassPre)
        # print("aggClassEst:",aggClassEst)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(Labels).T,np.ones((m,1))) #预测错误矩阵
        aggRate = np.sum(aggErrors)/m #错误率
        # print("aggRate:",aggRate)
        if aggRate == 0.00:
            break
    # return weakClassArr
    return weakClassArr,aggClassEst


####7.5 测试算法：基于adaboost的分类##################################################
def adaClassify(data,classfierArray): #adaboost分类函数
    #data,classfierArray分别表示待分类数据和多个弱分类器组成的数组
    dataMat = np.mat(data)
    m,n = dataMat.shape
    preLabel = np.mat(np.zeros((m,1)))

    for i in classfierArray:
        preScore = stumpClassify(dataMat,i['dim'],i['thresh'],i['ineqal'])
        preLabel += i['alpha'] * preScore
        # print("preLabel:",preLabel)
    return np.sign(preLabel)


####7.6 示例：在一个难数据集上应用adaboost##################################################
##自适应数据加载函数
def loadDataSet(fileName):
    dataArr = []
    label = []
    with open(fileName) as f1:
        content = f1.readlines()
        m = len(content) #数据条数
        for line in content:
            line = list(map(float,line.strip("\n").split("\t")))
            dataArr.append(line[0:-1])
            label.append(line[-1])
    return np.mat(dataArr),label


#####7.7 ROC曲线的绘制#############################################################################
#ROC曲线x轴代表false positive rate(FPR = FP/(FP+FN)) y轴表示true positive rate(TPR = TP/(TP+FN))
def plotROC(preScore,labels): #preScore,labels分别表示分类器的预测强度和实际标签
    cur = (1.0,1.0) #光标位置
    ySum = 0.0 #用于计算AUC的值
    m = len(labels) #样本个数
    num_positive = np.sum(np.mat(labels) == 1.0) #正例样本个数
    # print("num_positive:",num_positive)
    yStep = 1/float(num_positive) #y轴步长
    xStep = 1/float(m - num_positive) #x轴步长
    sortedIndices = preScore.argsort().tolist() #从小到大排序，返回索引

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)

    for i in sortedIndices[0]:
        if labels[i] == 1.0: #标签为1，在y轴倒退一个步长
            delX = 0.0
            delY = yStep
        else:#标签为-1，在x轴倒退一个步长
            delX = xStep
            delY = 0
            ySum += cur[1]
        # 在当前点cur和新点(cur[0] - delX,cur[1] - delY)之间画出一条线段
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    # 计算AUC需要对多个小矩形的面积进行累加，这些小矩形的宽度都是xStep，
    # 因此可对所有矩形的高度进行累加，然后再乘以xStep得到其总面积
    print("the Area Under the Curve is: ", ySum * xStep)




if __name__ == "__main__":
    ####7.3 基于单层决策树构建弱分类器##################################################
    # dataMat,labels = loadSimpData()
    # D = np.mat(np.ones((5,1))/5) #权重矩阵
    # bestStump, minError, bestClassPre = buildStump(dataMat,labels,D)

    ####7.4 完整adaboost算法实现##################################################
    # classfierArray = adaBoostTrainDS(dataMat,labels,9)
    # preLabel = adaClassify([[5,5],[0,0]],classfierArray)
    # print(preLabel)

    # ####7.6 示例：在一个难数据集上应用adaboost##################################################
    # dataMat,labels = loadDataSet("horseColicTraining2.txt")
    # classfierArray = adaBoostTrainDS(dataMat, labels, 10)
    # testMat,testLabel = loadDataSet("horseColicTest2.txt")
    # preLabel = adaClassify(testMat, classfierArray)
    # error = np.mat(np.ones((67,1)))
    # error[preLabel == np.mat(testLabel).T] = 0
    # print(error.sum())

    #####7.7 ROC曲线的绘制#############################################################################
    dataMat,labels = loadDataSet("horseColicTraining2.txt")
    classfierArray,preScore = adaBoostTrainDS(dataMat, labels, 10)
    plotROC(preScore.T,labels)
