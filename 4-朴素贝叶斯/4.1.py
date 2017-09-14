import numpy as np

######################1. 从文本中构建词向量########################################
def loadDataSet():
    postingList = [
        ['my','dog','has','flea','problems','help','please'],
        ['maybe','not','take','him','to','dog','park','stupid'],
        ['my','dalmation','is','so','cute','I','love','him'],
        ['stop','posting','stupid','worthless','garbage'],
        ['mr','licks','ate','my','steak','how','to','stop','him'],
        ['quit','buying','worthless','dog','food','stupid']
    ]
    label = [0,1,0,1,0,1] #1代表侮辱性文字 0代表正常言论
    return postingList,label

#找到所有出现在文档中的词，用以作为每一列特征
def createVocabList(lists):
    vocabList = set()
    for list0 in lists:
        vocabList = vocabList | set(list0)
    return list(vocabList)

#生成文档向量,文档词集模型
def setOfWord2Vec(vocablist, inputSet):
    returnVec = [0] * len(vocablist)
    for word in inputSet:
        if word in vocablist:
            returnVec[vocablist.index(word)] = 1
    return returnVec

################2. 从词向量计算概率###################################################
#朴素贝叶斯分类器训练函数  返回p(w/c(i)):类别为i时，各个词w出现的概率  p(c(i))：类别i的概率
def trainNB0(train_doc,labels):#train_docu表示文档向量 labels:表示文档的标签
    num_doc = len(train_doc) #文档个数
    word_num = len(train_doc[0]) #词的个数
    p1 = float(np.sum(labels))/float(num_doc) #类别为1的文档概率
    num_c0 = np.ones(word_num) #类别为0的文档中，各个词出现的次数初始化为1
    num_c1 = np.ones(word_num) #类别为1的文档中，各个词出现的次数
    num0 = 2 #类别为0的文档的总词数
    num1 = 2 #类别为1的文档的总词数

    for i in range(num_doc):
        if labels[i] == 1:
            num_c1 += train_doc[i]
            num1 += np.sum(train_doc[i])
        else:
            num_c0 += train_doc[i]
            num0 += np.sum(train_doc[i])
    p_c0 = [np.log(float(i)/float(num0)) for i in num_c0]
    p_c1 = [np.log(float(i) / float(num1)) for i in num_c1]
    return p_c0,p_c1,p1

#朴素贝叶斯分类函数
def classifyNB(testData,p_c0,p_c1,p_class1):
    p1 = np.dot(p_c1,testData) + np.log(p_class1)
    p0 = np.dot(p_c0, testData) + np.log(1-p_class1)
    if p1 > p0: return 1
    else: return 0

#便利函数
def testingNB():
    postingList,labels = loadDataSet()
    vocabList = createVocabList(postingList)
    train_doc = []
    for i in postingList:
        train_doc.append(setOfWord2Vec(vocabList, i))
    p_c0, p_c1, p1 = trainNB0(train_doc, labels)

    testEntry = ['love','my','dalmation']
    testVec = setOfWord2Vec(vocabList,testEntry)
    pre_label = classifyNB(testVec,p_c0, p_c1, p1)
    print("pre_label: ", pre_label)
    testEntry = ['stupid','garbage']
    testVec = setOfWord2Vec(vocabList, testEntry)
    pre_label = classifyNB(testVec, p_c0, p_c1, p1)
    print("pre_label: ", pre_label)

###############3. 文档词袋模型#####################################
def bagOfWords2Vec(vocablist,inputSet):
    returnVec = [0] * len(vocablist)
    for word in inputSet:
        if word in vocablist:
            returnVec[vocablist.index(word)] += 1
    return returnVec

##############4. 使用朴素贝叶斯过滤垃圾邮件##########################################
#将字符串解析为字符串列变
def textParse(bigString):
    import re
    regExt = re.compile(r"\W*")
    listOfToken = regExt.split(bigString)
    return [token.lower() for token in listOfToken if len(token) > 2]

#利用贝叶斯分类器进行分类，并输出错误率
def spamTest():
    import os
    textList = []
    labels = []
    #将每个文档处理成词列表
    for _, _, files in os.walk(os.path.dirname("C:\\zhoulili\\machinelearninginaction\\Ch04\\email\\spam\\")):
        for file in files:
            filePath = "C:\\zhoulili\\machinelearninginaction\\Ch04\\email\\spam\\" + file
            textList.append(textParse(open(filePath).read()))
            labels.append(1)
    for _, _, files in os.walk(os.path.dirname("C:\\zhoulili\\machinelearninginaction\\Ch04\\email\\ham\\")):
        for file in files:
            filePath = "C:\\zhoulili\\machinelearninginaction\\Ch04\\email\\ham\\" + file
            textList.append(textParse(open(filePath).read()))
            labels.append(0)
    vocabList = createVocabList(textList)
    #训练集、测试集
    testList = []
    testLabel = []
    for i in range(10):
        index = np.random.randint(0,len(textList))
        testList.append(textList[index])
        testLabel.append(labels[index])
        del textList[index]
        del labels[index]

    #生成训练集的词向量
    trainData = []
    for ele in textList:
        trainData.append(bagOfWords2Vec(vocabList,ele))

    p_c0, p_c1, p1 = trainNB0(trainData, labels)
    error_num = 0
    for i in range(10):
        testVec = bagOfWords2Vec(vocabList, testList[i])
        if classifyNB(testVec,p_c0, p_c1, p1) != testLabel[i]:
            error_num += 1
    print("error_rate: " , float(error_num)/float(len(testList)))





if __name__ == "__main__":
    ######################1. 从文本中构建词向量########################################
    # postingList,label = loadDataSet()
    # vocabList = createVocabList(postingList)
    # print(vocabList)
    # print(setOfWord2Vec(vocabList, postingList[0]))

    ################2. 从词向量计算概率###################################################
    # postingList, labels = loadDataSet()
    # vocabList = createVocabList(postingList)
    # train_doc = []
    # for i in postingList:
    #     train_doc.append(setOfWord2Vec(vocabList, i))
    # p_c0, p_c1, p1 = trainNB0(train_doc, labels)
    # for i in range(len(vocabList)):
    #     print(vocabList[i],p_c0[i],p_c1[i])
    #####################################
    # testingNB()

    ##############4. 使用朴素贝叶斯过滤垃圾邮件##########################################
    spamTest()