import numpy as np
import matplotlib.pyplot as plt

#加载数据集
def loadDataSet(fileName):
    fr = open(fileName).readlines()
    data = [list(map(float,line.strip("\n").split("\t"))) for line in fr]
    return np.mat(data)

#生成叶结点
def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])

#误差估计函数
def regErr(dataSet):
    return np.var(dataSet[:,-1])* dataSet.shape[0]

#将数据集以feature=value划分为两部分
def binSplitDataSet(dataSet,feature,value):
    data0 = dataSet[np.nonzero(dataSet[:,feature] > value),:][0]
    data1 = dataSet[np.nonzero(dataSet[:,feature] <= value),:][0]
    return data0,data1

#找到最优化分,进行了预剪枝
def chooseBestSplit(dataSet,leafType,errType,ops = (1,4)):
    tolS = ops[0] #容许的误差
    tolN = ops[1] #切分的最小样本数

    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #不进行划分
        return None,leafType(dataSet)

    S = errType(dataSet)
    bestS = np.inf
    bestFeature = 0
    bestValue = 0

    m,n = dataSet.shape
    for feature in range(n-1):
        for value in set(dataSet[:,feature].T.tolist()[0]):
            data0,data1 = binSplitDataSet(dataSet,feature,value)

            if data0.shape[0] < tolN or data1.shape[0] < tolN: continue #切分的样本数小于阀值，则不划分

            newS = errType(data0) + errType(data1)
            if newS < bestS:
                bestS = newS
                bestFeature = feature
                bestValue = value

    if (S - bestS) < tolS: #误差的减少不大，则不划分
        return None,leafType(dataSet)

    data0,data1 = binSplitDataSet(dataSet,bestFeature,bestValue)
    if data0.shape[0] < tolN or data1.shape[0] < tolN:
        return None,leafType(dataSet)
    return bestFeature,bestValue

#创建树
def createTree(dataSet,leafType = regLeaf, errType = regErr, ops = (1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None: return val #创建叶子结点

    retTree = {}
    retTree["feat"] = feat
    retTree["val"] = val
    data0,data1 = binSplitDataSet(dataSet,feat,val)
    retTree["left"] = createTree(data0,leafType,errType,ops)
    retTree["right"] = createTree(data1,leafType,errType,ops)
    return retTree

#############2 剪枝函数部分 ################################################################
#判断是否为叶子结点
def isTree(tree):
    return (type(tree).__name__ == "dict")

#从上往下遍历树，直至遇到叶子结点，如果遇到两个叶子结点，返回叶子结点的平均值
def getMean(tree):
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    return (tree['left']+tree['right']/2)

#剪枝函数
def prune(tree,testData):
    if testData.shape[0] == 0: return getMean(tree) #无测试数据时，对树做塌陷处理
    if isTree(tree['left']) or isTree(tree['right']): #不是叶子结点
        data0,data1 = binSplitDataSet(testData,tree['feat'],tree['val'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'],data0) #剪枝左子树
    if isTree(tree['right']): tree['right'] = prune(tree['right'],data1) #剪枝右子树

    if not isTree(tree['left']) and not isTree(tree['right']): #左右子树都是叶子结点
        data0, data1 = binSplitDataSet(testData, tree['feat'], tree['val'])
        errnoMerge = np.sum(np.power(data0[:, -1] - tree['left'],2)) + np.sum(np.power(data0[:, -1] - tree['left'],2))
        treeMean = tree['left']+tree['right']/2
        errMerge = np.sum(np.power(testData[:, -1] - treeMean,2))
        if errMerge < errnoMerge:
            print("merging")
            return treeMean
        else: return tree
    else: return tree

#######3 模型树部分 ##########################################################################
#线性模型
def linearSolve(dataSet):
    m,n = dataSet.shape
    mat1 = np.ones((m,1))
    x = np.hstack((mat1,dataSet[:,0:n-1]))
    y = dataSet[:,n-1]
    if np.linalg.det(x.T * x) != 0.00:
        w = ((x.T * x).I) * (x.T * y)
        return w,x,y
    else: return False

#叶子节点，应当返回线性模型
def modelLeaf(dataSet):
    w,x,y = linearSolve(dataSet)
    return w

#误差函数
def modelErr(dataSet):
    w,x,y = linearSolve(dataSet)
    y_hat = x * w
    return np.sum(np.power(y - y_hat,2))

#########4 树回归与标准回归的比较################################################################
#回归树叶子结点的预测
def regTreeEval(model,testData):
    return float(model)

#模型树叶子结点的预测
def modelTreeEval(model,testData):
    mat1 = np.ones((1, 1))
    x = np.hstack((mat1, testData[:,:]))
    return float(x*model)

#对单个测试样本进行预测
def treeForeCast(tree,testData,modelEval = regTreeEval):
    if not isTree(tree): return modelEval(tree,testData)

    if testData[tree['feat']] > tree['val']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],testData,modelEval)
        else:
            return modelEval(tree['left'],testData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],testData,modelEval)
        else: return modelEval(tree['right'],testData)

#对测试集进行预测
def createForeCast(tree,testData,modelEval = regTreeEval):
    m = testData.shape[0] #sample
    y_hat = np.zeros((m,1))
    for i in range(m):
        y_hat[i] = treeForeCast(tree,testData[i,:],modelEval)
    return np.mat(y_hat)

if __name__ == "__main__":

    ##########1.1回归树的创建    单次切分的例子####################################################################
    # ex00 = loadDataSet(".//machinelearninginaction//ch09//ex00.txt")
    # plt.scatter(ex00[:,0].T.tolist(),ex00[:,1].T.tolist())
    # plt.show()
    # retTree = createTree(ex00)
    # print(retTree)
    ##########1.2 回归树的创建  多次切分的例子###################################################################
    # ex0 = loadDataSet(".//machinelearninginaction//ch09//ex0.txt")
    # plt.scatter(ex0[:,1].T.tolist(),ex0[:,2].T.tolist())
    # plt.show()
    # retTree = createTree(ex0)
    # print(retTree)
    ###########2  回归树后剪枝 ###############################################################################
    # ex2 = loadDataSet(".//machinelearninginaction//ch09//ex2.txt")
    # plt.scatter(ex2[:,0].T.tolist(),ex2[:,1].T.tolist())
    # plt.show()
    # retTree = createTree(ex2,ops=(0,1))
    # ex2test = loadDataSet(".//machinelearninginaction//ch09//ex2test.txt")
    # mergeTree = prune(retTree,ex2test)
    # print(mergeTree)
    ###########3 模型树 #######################################################################################
    # exp2 = loadDataSet(".//machinelearninginaction//ch09//exp2.txt")
    # retTree = createTree(exp2,leafType = modelLeaf, errType = modelErr, ops = (1,10))
    # print(retTree)
    ############4 树回归与标准回归的比较 #####################################################################
    #4.1 回归树
    trainData = loadDataSet(".//machinelearninginaction//ch09//bikeSpeedVsIq_train.txt")
    testData = loadDataSet(".//machinelearninginaction//ch09//bikeSpeedVsIq_test.txt")
    myTree1 = createTree(trainData,ops = (1,20))
    y_hat1 = createForeCast(myTree1,testData[:,0],modelEval=regTreeEval)
    corr1 = np.corrcoef(y_hat1,testData[:,1],rowvar=0)[0,1]
    print("corr1: ",corr1)
    #4.2 模型树
    myTree2 = createTree(trainData,leafType=modelLeaf,errType=modelErr,ops = (1,20))
    y_hat2 = createForeCast(myTree2,testData[:,0],modelEval=modelTreeEval)
    corr2 = np.corrcoef(y_hat2, testData[:,1],rowvar=0)[0,1]
    print("corr2: ",corr2)
    w,x,y = linearSolve(trainData)
    print(w)
    ############################################################################