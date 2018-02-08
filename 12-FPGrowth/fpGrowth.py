# @Time    : 2018/2/8 下午3:20
# @Author  : zll
# @Site    : 
# @File    : fpGrowth.py
# @Software: PyCharm

####12.2 构建FP树#######################################################
##FP树的类定义
class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self,numOccur):
        self.count += numOccur

    def disp(self,ind = 1):#以文本形式显示树
        print(" "*ind,self.name," ",self.count)
        for child in self.children.values():
            child.disp(ind+1)

##使用数据集以及最小支持度，构建FP树
def createTree(dataSet,minSup = 1):
    headerTable = {} #头指针表
    for trans in dataSet:#首次遍历数据集，计数
        for item in trans:
            headerTable[item] = headerTable.get(item,0) + dataSet[trans]

    for key in list(headerTable.keys()): #删除非频繁项
        if headerTable[key] < minSup:
            del(headerTable[key])
    freqItemSet = set(headerTable.keys())

    if len(freqItemSet) == 0:return None,None
    for key in headerTable:
        headerTable[key] = [headerTable[key],None]

    retTree = treeNode("NUll Set",1,None)
    for trans,count in dataSet.items():
        localD = {}
        for item in trans:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0]for v in sorted(localD.items(),key=lambda k:k[1],reverse=True)] #根据全局频率对事务中的元素进行排序
            updateTree(orderedItems,retTree,headerTable,count) #FP树填充
    return retTree,headerTable

def updateTree(items,inTree,headerTable,count):#FP树填充
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count) #已存在该子节点，更新计数值
    else:
        inTree.children[items[0]] = treeNode(items[0],count,inTree)#不存在该子节点，新建树节点
        if headerTable[items[0]][1] == None:#更新头指针表
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1],inTree.children[items[0]])
    if len(items) > 1: #填充剩下的元素
        updateTree(items[1::],inTree.children[items[0]],headerTable,count)


def updateHeader(nodeOri,targetNode): ##更新头指针表
    while nodeOri.nodeLink != None:
        nodeOri = nodeOri.nodeLink
    nodeOri.nodeLink = targetNode

##构建简单数据集
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

##将数据集从列表转换为字典
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


####12.3 从FP树挖掘频繁项集#######################################################
##抽取条件模式基
def ascendTrees(leafNode,prefixPath):#迭代上溯FP树 leafNode表示与元素关联的头指针表，prefixPath存放上溯过程中遇到的节点
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTrees(leafNode.parent,prefixPath)

def findPrefixPath(basePat,treeNode):#找到以元素basePat结尾的所有前缀路径，treeNode表示与元素basePat关联的头指针表
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTrees(treeNode,prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

##递归查找频繁项
def mineTree(inTree,headerTable,minSup,preFix,freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(),key=lambda  x:x[1][0])] #从小到大排序
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat,headerTable[basePat][1]) #前缀路径
        myCondtree,myHead = createTree(condPattBases,minSup) #FP条件树

        if myHead != None:
            myCondtree.disp()
            mineTree(myCondtree,myHead,minSup,newFreqSet,freqItemList)



if __name__ == "__main__":
    ####12.2 构建FP树#######################################################
    # rootNode = treeNode("pyramid",9,None) #创建单节点
    # rootNode.children['eye'] = treeNode("eye",13,None) #rootNode的子节点
    # rootNode.children['phoenix'] = treeNode("phoenix", 3, None)  # rootNode的子节点
    # rootNode.disp()

    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    myFPtree,headerTable = createTree(initSet,3)
    myFPtree.disp()

    ####12.3 从FP树挖掘频繁项集#######################################################
    # print(findPrefixPath("x",headerTable["x"][1]))
    # print(findPrefixPath("z", headerTable["z"][1]))
    # print(findPrefixPath("r", headerTable["r"][1]))

    freqItem = [] #频繁项集
    mineTree(myFPtree,headerTable,3,set([]),freqItem)
    print("freqItem:",freqItem)

