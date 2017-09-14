import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt

######1 计算给定数据集的香农熵  最后一列是数据的标签值######################################################
def calcShannonEnt(dataSet):
    m,n = dataSet.shape
    shannonEnt = 0.00
    labelSet = set(dataSet[:,-1].T.tolist()[0])
    for label in labelSet:
        num = len(dataSet[np.nonzero(dataSet[:,-1] == label),:][0].tolist()) #标签为label的样本数
        shannonEnt  -= ((float(num)/float(m)) * math.log(float(num)/float(m),2))
    return shannonEnt

#创建一个简单的数据集
def createDataSet():
    dataSet = np.mat([[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']])
    labels = ['no surfacing','flippers']
    return dataSet,labels

##2 按照给定特征划分数据集 ############################################################################
def splitDataSet(dataSet,feat_index,value):
    resultData = dataSet[np.nonzero(dataSet[:,feat_index] == value),:][0]
    delData = np.delete(resultData,feat_index,axis = 1)
    return delData

######3.选择最好的数据集划分方式#######################################################################
def chooseBestFeatureToSplit(dataSet):
    m,n = dataSet.shape
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.00
    bestFeature = -1
    for i in range(n-1): #特征索引
        newEntropy = 0.00
        feat_value = set(dataSet[:,i].T.tolist()[0])
        for value in feat_value:
            delData = splitDataSet(dataSet,i,value) #返回的是特征i=value,并且去除特征i所在的列的数据
            newEntropy += float(delData.shape[0])/float(m) * calcShannonEnt(delData)
        if (baseEntropy - newEntropy) > bestInfoGain:
            bestInfoGain = baseEntropy - newEntropy
            bestFeature = i
    return bestFeature

################4.递归构建决策树####################################################################
#返回出现次数最多的类标签
def majorityCnt(dataSet):
    most_label = Counter(dataSet[:,-1]).most_common(1)[0]
    return most_label

#创建决策树
def createTree(dataSet,labels):
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #类别完全相同，停止划分
        return dataSet[0,-1]
    if dataSet.shape[1] == 1: #遍历完所有属性
        return majorityCnt(dataSet)
    bestFeat = chooseBestFeatureToSplit(dataSet) #找到最好划分
    print(dataSet)
    print(bestFeat)
    bestFeatLabel = labels[bestFeat]
    del labels[bestFeat]
    myTree = {bestFeatLabel:{}}

    feat_value = set(dataSet[:,bestFeat].T.tolist()[0])
    for value in feat_value:
        labels_bk = labels[:]
        delData = splitDataSet(dataSet,bestFeat,value)
        myTree[bestFeatLabel][value] = createTree(delData,labels_bk)
    return myTree

##############5. 使用文本注解绘制树结点##################################################################
def plotNode(anno_text,xyPt,xyText,nodeType):
    arrow_arg = dict(arrowstyle="<-")  # 设置箭头格式
    # createPlot.ax1.annotate(anno_text, xy = xyPt, xytext = xyText, xycoords = 'data',textcoords = 'data', arrowprops = arrow_arg,
    #                         va = 'center',ha = 'left',bbox = nodeType)
    createPlot2.ax1.annotate(anno_text, xy=xyPt, xytext=xyText, xycoords='data', textcoords='data', arrowprops=arrow_arg,
                             va = 'center',ha = 'center',bbox = nodeType)

def createPlot():
    fig = plt.figure(1,facecolor='white')
    fig.clf() #清空绘图区

    decisionNode = dict(boxstyle='sawtooth', fc='0.8')  # 设置决策结点的文本框格式
    leafNode = dict(boxstyle='round4', fc='0.8')  # 设置叶子结点的文本框格式

    createPlot.ax1 = fig.add_subplot(111)
    plotNode('decisionNode',(0.1,0.5),(0.5,0.1),decisionNode) #决策结点
    plotNode('leafNode',(0.3,0.8),(0.8,0.3),leafNode) #叶子结点
    plt.show()

########6. 获取叶子结点的数目和树的层次##################################################################
#输出预先存储的树信息
def retrieveTree(i):
    listOfTress = [
        {'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},
        {'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}
    ]
    return listOfTress[i]

#获取叶子结点的数目
def getNumLeafs(myTree):
    numLeaf = 0
    firstKey = list(myTree.keys())[0]
    secondDict = myTree[firstKey]
    for ele in secondDict.keys():
        if type(secondDict[ele]).__name__ == 'dict': #不是叶子结点
            numLeaf += getNumLeafs(secondDict[ele])
        else:
            numLeaf += 1
    return numLeaf

#获取树的层数
def getTreedepth(myTree):
    maxDepth = 0
    firstKey = list(myTree.keys())[0]
    secondDict = myTree[firstKey]
    treeDepth = 0
    for ele in secondDict.keys():
        if type(secondDict[ele]).__name__ == 'dict': #不是叶子结点
            treeDepth += getTreedepth(secondDict[ele])
        else:
            treeDepth = 1
        if treeDepth > maxDepth:
            maxDepth = treeDepth
    return maxDepth

##############7. 绘制决策树###########################################################################
#在父子节点间添加文本信息
def plotMidtext(parentPt,cntrPt,nodeTxt):
    midX = (parentPt[0] + cntrPt[0]) / 2.0
    midY = (parentPt[1] + cntrPt[1]) / 2.0
    createPlot2.ax1.text(midX,midY,nodeTxt)


def plotTree(myTree,parentPt,nodeTxt):
    numLeaf = float(getNumLeafs(myTree))
    depth = float(getTreedepth(myTree))
    decisionNode = dict(boxstyle='sawtooth', fc='0.8')  # 设置决策结点的文本框格式
    leafNode = dict(boxstyle='round4', fc='0.8')  # 设置叶子结点的文本框格式

    firstKey = list(myTree.keys())[0]
    secondDict = myTree[firstKey]

    cntrPt = (plotTree.xOff + (1.0+numLeaf)/2.0/plotTree.totalW,plotTree.yOff) #当前结点坐标
    plotNode(firstKey,parentPt,cntrPt,decisionNode) #画parentPt节点
    plotMidtext(parentPt,cntrPt,nodeTxt) #添加parentPt,cntrPt之间的文本信息

    plotTree.yOff -= 1.0/plotTree.totalD

    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key],cntrPt,key)
        else:
            plotTree.xOff += 1.0/plotTree.totalW
            plotNode(secondDict[key], cntrPt, (plotTree.xOff,plotTree.yOff), leafNode)  # 画parentPt节点
            plotMidtext((plotTree.xOff,plotTree.yOff), cntrPt, key)  # 添加parentPt,cntrPt之间的文本信息

    plotTree.yOff += 1.0 / plotTree.totalD


def createPlot2(myTree):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    createPlot2.ax1 = fig.add_subplot(111)

    plotTree.totalW = float(getNumLeafs(myTree))
    plotTree.totalD = float(getTreedepth(myTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0

    plotTree(myTree,(0.5,1.0),"")
    plt.show()

##############8. 使用决策树执行分类#################################################################
def classify(myTree,featLabel,testVec):
    firstKey = list(myTree.keys())[0]
    secondDict = myTree[firstKey]

    featIndex = featLabel.index(firstKey)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                return classify(secondDict[key],featLabel,testVec)
            else:
                return secondDict[key]

########9. 使用pickle存储决策树###################################################################
import pickle
def storeTree(mytree,file):
    fr = open(file,'wb')
    pickle.dump(mytree,fr)
    fr.close()

def grabTree(file):
    fr = open(file,'rb')
    return pickle.load(fr)

#######################10. 使用决策树预测隐形眼镜类型#######################################################
def loadDataSet(file):
    fr = open(file).readlines()
    data = np.mat([list(line.strip("\n").split("\t")) for line in fr])
    return data

if __name__ == "__main__":
    ##########1.  计算香农熵##########################################################################
    # dataSet,labels = createDataSet()
    # Ent = calcShannonEnt(dataSet)
    # print(Ent)
    ########2 按照给定特征划分数据集 ###################################################################
    # print(splitDataSet(dataSet,0,'0'))
    ###############3. 选择最好的数据集划分方式 ########################################################
    # print(chooseBestFeatureToSplit(dataSet))
    ################4.递归构建决策树####################################################################
    # print(createTree(dataSet,labels))
    ##############5. 使用文本注解绘制树结点##################################################################
    # createPlot()
    ########6. 获取叶子结点的数目和树的层次##################################################################
    # myTree = retrieveTree(0)
    # print(getNumLeafs(myTree))
    # print(getTreedepth(myTree))
    ##############7. 绘制树###########################################################################
    # myTree = retrieveTree(0)
    # createPlot2(myTree)
    ##############8. 使用决策树执行分类#################################################################
    # dataSet,labels = createDataSet()
    # myTree = retrieveTree(0)
    # print(classify(myTree,labels,[1,0]))
    # print(classify(myTree, labels, [1, 1]))
    ########9. 使用pickle存储决策树###################################################################
    # myTree = retrieveTree(0)
    # storeTree(myTree,"3.txt")
    # print(grabTree("3.txt"))
    ##################10. 使用决策树预测隐形眼镜类型#########################
    data = loadDataSet(".//machinelearninginaction//ch03//lenses.txt")
    labels = ['age','prescript','astigmatic','tearRate']
    myTree = createTree(data,labels)
    createPlot2(myTree)