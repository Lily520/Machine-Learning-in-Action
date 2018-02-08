# @Time    : 2018/2/8 上午9:17
# @Author  : zll
# @Site    : 
# @File    : apriori.py.py
# @Software: PyCharm

###11.3 使用Apriori算法来发现频繁集######################################################################
##创建一个用于测试的简单数据集
def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

##构建集合C1(大小为1的所有候选项集)
def createC1(dataSet):
    C1 = []

    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    # print(set(map(frozenset,C1)))
    return set(map(frozenset,C1))

##从C1生成L1(大小为1的所有频繁集)
def scanD(D,Ck,minSupport):#D为数据集，Ck为候选集，minSupport为最小支持度
    ssCnt = {} #存放候选项的出现次数
    numItems = len(D) #数据集大小
    retList = [] #存放频繁项
    supportData = {} #存放频繁项集支持度

    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt.keys():
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1

    for key in ssCnt.keys():
        support = float(ssCnt[key])/float(numItems)
        if support >= minSupport:
            retList.append(key)
        supportData[key] = support

    return retList,supportData

##Apriori
def aprioriGen(Lk,k): #创建候选集Ck
    ##Lk为频繁项，k为项集元素个数
    retlist = []
    for i in range(len(Lk)-1):
        for j in range(i+1,len(Lk)):
            L1 = list(Lk[i])[:k-2]
            L1.sort()
            L2 = list(Lk[j])[:k-2]
            L2.sort()
            if L1 == L2:
                retlist.append((Lk[i] | Lk[j])) #集合并
    return retlist



def apriori(dataSet,minSupport=0.5):
    C1 = createC1(dataSet)
    D = list(map(set,dataSet))
    L1,supportData = scanD(D,C1,minSupport)
    L = [L1]
    k = 2

    while len(L[k-2]) > 0:
        Ck = aprioriGen(L[k-2],k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1

    return L,supportData


###11.4 从频繁项挖掘关联规则######################################################################

def generateRules(L,supportData,minConf = 0.7):
    bigRuleList = []
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet] #对每个频繁项集，创建只包含单个元素的列表
            if i > 1:#频繁项集元素超过2，进行合并
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:#频繁项集元素为2，直接计算可信度
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList

def calcConf(freqSet,H,supportData,bigRuleList,minConf):#计算置信度
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet-conseq,"-->",conseq,"   conf:",conf)
            bigRuleList.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet,H,supportData,bigRuleList,minConf): #合并
    #H表示出现在规则右部的元素列表
    m = len(H[0])
    if len(freqSet) > (m+1):
        Hmp1 = aprioriGen(H,m+1) #创建规则右边长度为(m+1)的候选项
        Hmp1 = calcConf(freqSet,Hmp1,supportData,bigRuleList,minConf)
        if len(Hmp1) > 1: #如果不止一条规则满足, 则尝试进一步合并规则右边
            rulesFromConseq(freqSet, Hmp1, supportData, bigRuleList, minConf)






if __name__ == "__main__":
    ###11.3 使用Apriori算法来发现频繁集######################################################################
    dataSet = loadDataSet()
    L,supportData = apriori(dataSet)
    # print(supportData)
    generateRules(L,supportData,minConf=0.5)