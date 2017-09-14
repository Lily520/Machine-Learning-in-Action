from bs4 import BeautifulSoup
import numpy as np

#岭回归
def ridgeRegres(x,y,lam=0.2):
    m = x.shape[1]

    E = np.eye(m) #单位阵

    # print(np.dot((x.T),x))
    x_trans = np.mat(np.dot((x.T),x) + lam * E)
    if np.linalg.det(x_trans) != 0.00:
        w_hat = x_trans.I * (x.T) * y
        return w_hat.T
    else:
        return False

def ridgeTest(x,y):
    x_norm = (x - np.mean(x, axis=0)) / np.var(x, axis=0)  # (x-mean(x))/var(x)
    y_norm = y - np.mean(y, axis=0)
    m = x.shape[1]
    num_iter = 30
    w_mat = np.mat(np.zeros((num_iter,m)))

    for i in range(num_iter):
        w_mat[i,:] = ridgeRegres(x_norm,y_norm,np.exp(i-10))
    return w_mat

#误差
def error(y,y_hat):
    return (np.multiply((y-y_hat),(y-y_hat))).sum()


#1.购物信息获取函数
def searchForSet(x,y,html,year,numPiece,ori_price):
    soup = BeautifulSoup(open(html,encoding="utf-8"),from_encoding="utf-8")

    for ele in soup.find_all("table",class_ = "li"):
        sold_tag = ele.find("span",class_ = "sold")
        if sold_tag != None: #只获取有已出售标志的产品

            pro_name = ele.find("div",class_ = "ttl").a.get_text() #获取产品名称
            if "new" in pro_name.lower() or "nisb" in pro_name.lower(): #new_flag 新产品标志
                new_flag = 1
            else:
                new_flag = 0
            price = float(ele.find_all("td")[4].get_text().replace("$","").replace("Free shipping","").replace(",",""))
            #只保留成套产品的数据  将产品数据存放到x,y中
            if price > ori_price*0.5:
                x.append([year,numPiece,new_flag,ori_price])
                y.append(price)

##2.交叉验证测试岭回归
def crossValidation(data,numVal=10): #numVal交叉验证次数
    m,n = data.shape
    trainNum = int(m * 0.9) #90%做训练集
    errorMat = np.zeros((numVal,30)) #误差矩阵
    print(errorMat.shape)

    for i in range(numVal):
        np.random.shuffle(data) #打乱
        #划分训练集和测试集
        trainData = data[0:trainNum,:]
        testData = data[trainNum+1:,:]
        #训练模型
        w = ridgeTest(trainData[:,0:-1],trainData[:,-1]) #w.shape=(30,4)

        #标准化测试数据集
        testx_norm = (testData[:,0:-1] - np.mean(trainData[:,0:-1],axis=0))/np.var(trainData[:,0:-1],axis=0) #(65,4)
        testy_norm = (testData[:,-1] - np.mean(trainData[:,-1],axis=0))/(np.var(trainData[:,-1],axis=0)) #(65,1)

        for j in range(30): #30组回归系数
            y_hat = testx_norm * w[j,:].T + np.mean(trainData[:,-1],axis=0)
            errorMat[i,j] = error(testy_norm, y_hat)

    meanError = np.mean(errorMat,axis=0) #每组平均误差
    minMean = min(meanError)
    bestWeights = w[np.nonzero(meanError == minMean)] #最优回归系数

    #回归系数还原
    bestw_origin = bestWeights / (np.var(data[:,0:-1],axis=0)) #(1,4)
    constant = np.sum(np.multiply(np.mean(data[:,0:-1],axis=0),bestw_origin)) + np.mean(data[:,-1],axis=0) #常数项
    return bestw_origin,constant


if __name__ == "__main__":
    # html0 = open("C:\\zhoulili\\machinelearninginaction\\Ch08\\setHtml\\lego8288.html", encoding="utf-8")
    # html1 = open("C:\\zhoulili\\machinelearninginaction\\Ch08\\setHtml\\lego10030.html", encoding="utf-8")
    # html2 = open("C:\\zhoulili\\machinelearninginaction\\Ch08\\setHtml\\lego10179.html", encoding="utf-8")
    # html3 = open("C:\\zhoulili\\machinelearninginaction\\Ch08\\setHtml\\lego10181.html", encoding="utf-8")
    # html4 = open("C:\\zhoulili\\machinelearninginaction\\Ch08\\setHtml\\lego10189.html", encoding="utf-8")
    # html5 = open("C:\\zhoulili\\machinelearninginaction\\Ch08\\setHtml\\lego10196.html", encoding="utf-8")

    ############ 1.购物数据获取 ########################################################
    lgX = []
    lgY = []
    html_list = [8288,10030,10179,10181,10189,10196]
    year_list = [2006,2002,2007,2007,2008,2009]
    numPiece = [800,3096,5195,3428,5922,3263]
    ori_price = [49.99,269.99,499.99,199.99,299.99,249.99]
    for i in range(6):
        html = "C:\\zhoulili\\machinelearninginaction\\Ch08\\setHtml\\lego" + str(html_list[i]) + ".html"
        searchForSet(lgX,lgY,html,year_list[i],numPiece[i],ori_price[i])

    lgX = np.mat(lgX)
    # x_0 = np.ones((lgX.shape[0],1))
    # lgX = np.hstack((x_0,lgX))
    lgY = np.mat(lgY).T
    data = np.hstack((lgX,lgY))
    # print(type(data))
    ##################################################################################
    #2.交叉验证测试岭回归
    w,constant = crossValidation(data) #w=(1,4)
