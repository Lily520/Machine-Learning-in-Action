import numpy as np


#############6-1 SMO算法中的辅助函数#############################################################
def loadDataSet(fileName): #加载数据集，返回列表
    content = open(fileName).readlines()
    data = [list(map(float,line.strip("\n").split("\t")[0:-1])) for line in content] #特征
    label = [float(line.strip("\n").split("\t")[-1]) for line in content] #标签
    return data,label

def selectJrand(i,m): #选择alpha[j]
    j = i
    while j == i:
        j = np.random.randint(0,m)
    return j

def clipAlpha(alphaj,H,L): #调整alphaj的范围在(L,H)之间
    if alphaj > H: alphaj = H
    if alphaj < L: alphaj = L
    return alphaj

####################6-2 简化版SMO算法#############################################################
def smoSimple(data,label,C,toler,maxIter):

    dataMat = np.mat(data)
    labelMat = np.mat(label).T
    m = dataMat.shape[0] #数据条数
    alpha = np.mat(np.zeros((m,1))) #初始化alpha矩阵
    b = 0.00
    iter = 0
    while iter < maxIter: #iter记录的是没有任何alpha改变的情况下，遍历数据集的次数
        alphaPairsChanged = 0 #一次循环中alpha是否已经进行优化
        for i in range(m):
            fxi=float(dataMat[i,:]*(dataMat.T)*np.multiply(alpha,labelMat))+b #样本i的预测值
            Ei=fxi - float(labelMat[i]) #误差值
            if (labelMat[i]*Ei < -toler and alpha[i] < C) or (labelMat[i]*Ei > toler and alpha[i] > 0):
                j = selectJrand(i,m) #选择第二个alpha
                fxj = float(dataMat[j, :] * (dataMat.T) * np.multiply(alpha, labelMat)) + b  # 样本j的预测值
                Ej = fxj - float(labelMat[j])  # 误差值
                alpha_oldi = alpha[i].copy()
                alpha_oldj = alpha[j].copy()

                #设置H,L的值
                if labelMat[i] != labelMat[j]:
                    L = max(0,alpha[j]-alpha[i])
                    H = min(C,C+alpha[j]-alpha[i])
                else:
                    L = max(0,alpha[j]+alpha[i]-C)
                    H = min(C,alpha[j]+alpha[i])
                if L == H: print("L==H");continue

                eta=dataMat[i,:]*dataMat[i,:].T+dataMat[j,:]*dataMat[j,:].T-2*dataMat[i,:]*dataMat[j,:].T
                if eta <= 0:print("eta<=0");continue

                alpha[j] = alpha_oldj + labelMat[j]*(Ei-Ej)/eta #更新alpha[j]
                alpha[j] = clipAlpha(alpha[j],H,L)
                if(abs(alpha[j]-alpha_oldj) < 0.00001):
                    print("j is not movinf enough!")
                    continue

                alpha[i] = alpha_oldi + labelMat[i]*labelMat[j]*(alpha_oldj-alpha[j]) #更新alpha[i]
                #更新b
                bi=b-Ei-labelMat[i]*(alpha[i]-alpha_oldi)*dataMat[i,:]*dataMat[i,:].T-labelMat[j]*(alpha[j]-alpha_oldj)*dataMat[i,:]*dataMat[j,:].T
                bj=b-Ej-labelMat[i]*(alpha[i]-alpha_oldi)*dataMat[i,:]*dataMat[j,:].T-labelMat[j]*(alpha[j]-alpha_oldj)*dataMat[j,:]*dataMat[j,:].T
                if alpha[i] > 0 and alpha[i] < C: b = bi
                elif alpha[j] > 0 and alpha[j] < C: b = bj
                else: b = (bi+bj)/2.0
                alphaPairsChanged += 1
                print("iter:",iter,"i: ",i,"alphaPairsChanged: ",alphaPairsChanged)

            if alphaPairsChanged == 0: iter += 1
            else: iter = 0
            print("iteration:",iter)
    return b,alpha

#################6-3 完整版SMO支持函数###############################################################
class optStruct:
    def __init__(self,dataMat,labelMat,C,toler,kTup): #kTup是核函数版本对应的参数
        self.X = dataMat
        self.label = labelMat
        self.C = C
        self.tol = toler
        self.m = dataMat.shape[0]
        self.alpha = np.mat(np.zeros((self.m,1)))
        self.b = 0.00
        self.eCache = np.mat(np.zeros((self.m,2))) #第一列标志位，第二列实际的误差值
        ##下面是核函数版本增加的参数
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup)


def calcEk(oS,k): #计算误差Ek
    # fxk = float(oS.X[k, :] * (oS.X.T) * np.multiply(oS.alpha, oS.label)) + oS.b  # 样本i的预测值 非核函数版本
    fxk = float(oS.K[:, k].T * np.multiply(oS.alpha, oS.label)) + oS.b  # 样本i的预测值 核函数版本
    Ek = fxk - float(oS.label[k])
    return Ek

def selectJ(i,oS,Ei): #选择最优的j,返回j和误差Ej
    maxj = -1
    maxDeltaj = 0
    Ej = 0.00

    oS.eCache[i] = [1,Ei]
    validEcacheList = np.nonzero(oS.eCache[:,0])[0] #第一列为有效标志位的行号

    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS,k)
            deltaE = abs(Ek - Ei)
            if deltaE > maxDeltaj: #选择具有最大步长的j
                maxDeltaj = deltaE
                maxj = k
                Ej = Ek
        return maxj,Ej
    else:
        j = selectJrand(i,oS.m)
        return j,calcEk(oS,j)

def updateEk(oS,k): #更新Ek
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]


##############6-4 完整SMO算法中的优化例程##############################################################
def innerL(i,oS):
    Ei = calcEk(oS,i)
    if (oS.label[i] * Ei < -oS.tol and oS.alpha[i] < oS.C) or (oS.label[i] * Ei > oS.tol and oS.alpha[i] > 0):
        j,Ej = selectJ(i,oS,Ei)  # 选择第二个alpha
        alpha_oldi = oS.alpha[i].copy()
        alpha_oldj = oS.alpha[j].copy()

        # 设置H,L的值
        if oS.label[i] != oS.label[j]:
            L = max(0, oS.alpha[j] - oS.alpha[i])
            H = min(oS.C, oS.C + oS.alpha[j] - oS.alpha[i])
        else:
            L = max(0, oS.alpha[j] + oS.alpha[i] - oS.C)
            H = min(oS.C, oS.alpha[j] + oS.alpha[i])
        if L == H: print("L==H"); return 0

        #eta = oS.X[i, :] * oS.X[i, :].T + oS.X[j, :] * oS.X[j, :].T - 2 * oS.X[i, :] * oS.X[j, :].T #非核函数版本
        eta = oS.K[i,i] + oS.K[j,j] - 2 * oS.K[i,j]  #核函数版本
        if eta <= 0: print("eta<=0");return 0

        oS.alpha[j] = alpha_oldj + oS.label[j] * (Ei - Ej) / eta  # 更新alpha[j]
        oS.alpha[j] = clipAlpha(oS.alpha[j], H, L)
        updateEk(oS,j) #更新误差缓存
        if (abs(oS.alpha[j] - alpha_oldj) < 0.00001):
            print("j is not moving enough!")
            return 0

        oS.alpha[i] = alpha_oldi + oS.label[i] * oS.label[j] * (alpha_oldj - oS.alpha[j])  # 更新alpha[i]
        # 更新b-非核函数版本
        # bi = oS.b - Ei - oS.label[i] * (oS.alpha[i] - alpha_oldi) * oS.X[i, :] * oS.X[i, :].T - oS.label[j] * (
        # oS.alpha[j] - alpha_oldj) * oS.X[i, :] * oS.X[j, :].T
        # bj = oS.b - Ej - oS.label[i] * (oS.alpha[i] - alpha_oldi) * oS.X[i, :] * oS.X[j, :].T - oS.label[j] * (
        # oS.alpha[j] - alpha_oldj) * oS.X[j, :] * oS.X[j, :].T

        #更新b 核函数版本
        bi = oS.b - Ei - oS.label[i] * (oS.alpha[i] - alpha_oldi) * oS.K[i,i] - oS.label[j] * (oS.alpha[j] - alpha_oldj) * oS.K[i,j]
        bj = oS.b - Ej - oS.label[i] * (oS.alpha[i] - alpha_oldi) * oS.K[i, j] - oS.label[j] * (oS.alpha[j] - alpha_oldj) * oS.K[j, j]

        if oS.alpha[i] > 0 and oS.alpha[i] < oS.C:
            oS.b = bi
        elif oS.alpha[j] > 0 and oS.alpha[j] < oS.C:
            oS.b = bj
        else:
            oS.b = (bi + bj) / 2.0
        return 1
    else:
        return 0


##############6-5 完整版SMO的外循环代码###############################################################
def smoP(dataArr,labelArr,C,toler,maxIter,kTup=('lin',0)):
    oS = optStruct(np.mat(dataArr),np.mat(labelArr).T,C,toler,kTup)
    print(oS.X.shape,oS.label.shape)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        if entireSet: #遍历整个数据集
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print("fullSet,iter: ",iter,"i: ",i,"alphaPairsChanged: ",alphaPairsChanged)
            iter += 1
        else: #遍历非边界值
            nonBounds = np.nonzero((oS.alpha.A > 0) * (oS.alpha.A < oS.C))[0]
            for i in nonBounds:
                alphaPairsChanged += innerL(i,oS)
                print("nonBound,iter: ", iter, "i: ", i, "alphaPairsChanged: ", alphaPairsChanged)
            iter += 1

        if entireSet: entireSet = False
        elif alphaPairsChanged == 0: entireSet = True
        print("iteration: ",iter)
    return oS.b,oS.alpha

def calcWs(alpha,label,data):
    dataMat = np.mat(data)
    labelMat = np.mat(label).T
    w = dataMat.T * np.multiply(alpha,labelMat)
    return w


####################6-6 核函数转换##################################################################
def kernelTrans(X,A,kTup):
    m,n = X.shape
    K = np.mat(np.zeros((m,1)))
    if kTup[0] == 'lin': K = X * A.T
    elif kTup[0] == 'rbf':
        for i in range(m):
            K[i] = (X[i,:] - A) * (X[i,:] - A).T
        K = np.exp(-K/(kTup[1]**2))
    else:
        raise NameError("that kernel is not recognized!")
    return K


####################6-8 利用核函数进行分类的径向基测试函数################################################
def testRbf(k1 = 1.3):
    dataArr,labelArr = loadDataSet("testSetRBF.txt")
    b, alpha = smoP(dataArr, labelArr, 200,0.0001,10000, ('rbf',k1))
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    #构建支持向量矩阵
    svInd = np.nonzero(alpha.A > 0)[0]
    sVs = dataMat[svInd] #支持向量
    labelSV = labelMat[svInd]
    print("there are ",sVs.shape[0]," Support Vectors.")

    m,n = dataMat.shape
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T * np.multiply(labelSV,alpha[svInd])+b
        if np.sign(predict) != labelMat[i]:
            errorCount += 1
    print("training error: ", float(errorCount)/float(m))

    #test set
    dataArr, labelArr = loadDataSet("testSetRBF2.txt")
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    m, n = dataMat.shape
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alpha[svInd]) + b
        if np.sign(predict) != labelMat[i]:
            errorCount += 1
    print("test error: ", float(errorCount) / float(m))

############6-9 基于SVM的手写数字识别#################################################################
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir

    fileNameList = listdir(dirName)
    m = len(fileNameList)
    trainX = np.zeros((m,1024))
    trainY = []
    for i in range(m):
        label = float(fileNameList[i].split("_")[0])
        if label != 9:
            filename = dirName + "/" + fileNameList[i]
            trainY.append(1)
            trainX[i,:] = img2vector(filename)
        elif label == 9:
            filename = dirName + "/" + fileNameList[i]
            trainY.append(-1)
            trainX[i, :] = img2vector(filename)
    return trainX,trainY

def testDigits(kTup=('rbf',10)):
    dataArr,labelArr = loadImages("digits/trainingDigits")
    b, alpha = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    # 构建支持向量矩阵
    svInd = np.nonzero(alpha.A > 0)[0]
    sVs = dataMat[svInd]  # 支持向量
    labelSV = labelMat[svInd]
    print("there are ", sVs.shape[0], " Support Vectors.")

    m, n = dataMat.shape
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alpha[svInd]) + b
        if np.sign(predict) != labelMat[i]:
            errorCount += 1
    print("training error: ", float(errorCount) / float(m))

    # test set
    dataArr, labelArr = loadImages("digits/testDigits")
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    m, n = dataMat.shape
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alpha[svInd]) + b
        if np.sign(predict) != labelMat[i]:
            errorCount += 1
    print("test error: ", float(errorCount) / float(m))


if __name__ == "__main__":
    ####################6-2 简化版SMO算法#############################################################
    # data,label = loadDataSet("testSet.txt")
    # b,alpha = smoSimple(data,label,0.6,0.001,40)
    # print(b)
    # print(alpha[alpha > 0])
    ##############6-5 完整版SMO的外循环代码###############################################################
    # data, label = loadDataSet("testSet.txt")
    # b, alpha = smoP(data, label, 0.6, 0.001, 40)
    # ws = calcWs(alpha,label,data)
    # print(ws)
    # dataMat = np.mat(data)
    # print("预测值：",dataMat[0]*np.mat(ws)+b)
    # print("实际值： ",label[0])
    ####################6-8 利用核函数进行分类的径向基测试函数################################################
    # testRbf()
    ############6-9 基于SVM的手写数字识别#################################################################
    testDigits()