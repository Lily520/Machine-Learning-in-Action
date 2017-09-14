import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

#可视化ex0.txt 画散点图和拟合线
def plot_scatter_line(x,y,y_hat):
    feature_x = [float(t[1]) for t in x]
    plt.figure()
    plt.scatter(feature_x,y) #散点图
    feature_mat = np.mat(feature_x).T
    com_x_yhat = np.hstack((feature_mat,y_hat)) #在行上合并矩阵
    com_sort = sorted(com_x_yhat.tolist(),key = lambda x:x[0]) #排序
    com_feature = [x[0] for x in com_sort]
    com_label = [x[1] for x in com_sort]
    print(com_sort)
    plt.plot(com_feature,com_label,color = "r") #拟合线
    plt.show()

def loadDataSet(file):
    with open(file,encoding="utf-8") as f1:
        content = f1.readlines()
        xArr = [x.split("\t")[:-1] for x in content]
        yArr = [float(x.split("\t")[-1].strip("\n")) for x in content]
    return xArr,yArr

#线性回归
def compute_w(testArr,x,y):
    x = (x - np.mean(x, axis=0)) / np.var(x, axis=0)  # (x-mean(x))/var(x)
    y = y - np.mean(y, axis=0)
    if np.linalg.det(x.T * x) != 0.00:
        w = ((x.T * x).I) * (x.T * y)
        y_hat = x * w
        print("compute_w:")
        print(w.T)
        return y_hat
    else:
        return False


#局部加权线性回归
def lwlr(testPoint, x, y, k = 1.0):
    m = x.shape[0] #样本个数
    W = np.zeros((m,m)) #全零方阵

    #求每一个样本的权重值
    for i in range(m):
        diff_X = testPoint - x[i]
        W[i][i] = np.exp(diff_X * (diff_X.T)/(-2*k**2))

    x_W = x.T * W * x
    if np.linalg.det(x_W) != 0.00:
        w_hat = x_W.I * (x.T) * W * y
        y_test = testPoint * w_hat
        return y_test
    else:
        return False

def lwlr_all(testArr,x, y, k = 1.0):
    m = x.shape[0]
    y_hat = np.zeros((m,1))
    for i in range(m):
        y_hat[i][0] = lwlr(testArr[i], x, y, k)
    return np.mat(y_hat)

#误差
def error(y,y_hat):
    return (np.multiply((y-y_hat),(y-y_hat))).sum()

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
    x_norm = preprocessing.scale(x)
    y_norm = preprocessing.scale(y)
    m = x.shape[1]
    num_iter = 30
    w_mat = np.mat(np.zeros((num_iter,m)))

    for i in range(num_iter):
        w_mat[i,:] = ridgeRegres(x_norm,y_norm,np.exp(i-10))
    return w_mat

#4.前向逐步回归
def StageWise(x,y,eps,numIt):
    x_norm = (x - np.mean(x,axis=0))/np.var(x,axis=0) # (x-mean(x))/var(x)
    y_norm = y-np.mean(y,axis=0)
    m,n = x_norm.shape
    ws = np.zeros((n,1))
    return_w = np.zeros((numIt,n))

    for i in range(numIt):
        # print(ws.T)
        lowestError = np.inf #无穷大
        for j in range(n):
            for sign in [-1,1]:
                ws_test = ws.copy()
                ws_test[j] += eps*sign
                y_test = x_norm * np.mat(ws_test)
                error_test = error(y_norm,y_test)
                if error_test < lowestError:
                    lowestError = error_test
                    w_best = ws_test
        return_w[i,:] = w_best.T
        ws = w_best
    return return_w


if __name__ == "__main__":
    # input_path = "C:\\zhoulili\\machinelearninginaction\\Ch08\\ex0.txt"
    input_path = "C:\\zhoulili\\machinelearninginaction\\Ch08\\abalone.txt"
    x,y = loadDataSet(input_path)
    x_mat = np.mat(x,dtype=float) #list转矩阵 并指定元素类型
    y_mat = np.mat(y,dtype=float).T

    ######################################################################################
    # #4.前向逐步回归
    # eps = 0.001
    # numIt = 5000
    # ws = StageWise(x_mat,y_mat,eps,numIt)
    # plt.plot(ws)
    # plt.show()
    # print(ws)
    # ##################################################################################
    # #3.岭回归
    # w_mat = ridgeTest(x_mat,y_mat)
    # print(w_mat)
    # plt.plot(w_mat)
    # plt.show()
    #
    ####################################################################################
    #1.线性回归
    # y_hat = compute_w(x_mat,x_mat, y_mat)
    # plot_scatter_line(x, y, y_hat)

    # #2. 局部加权线性回归  不同参数k 求得的拟合线
    # y_hat = lwlr_all(x_mat, y_mat, 1.0) #k=1.0
    # plot_scatter_line(x, y, y_hat)
    #
    # y_hat = lwlr_all(x_mat, y_mat, 0.01) #k=0.01
    # plot_scatter_line(x, y, y_hat)
    #
    # y_hat = lwlr_all(x_mat, y_mat,0.003) #k=0.03
    # plot_scatter_line(x, y, y_hat)
    ##############################################################################
    #鲍鱼数据上的实验结果
    # y_hat1 = lwlr_all(x_mat[0:99],x_mat[0:99],y_mat[0:99],k=0.1)
    # y_train01 = error(y_mat[0:99],y_hat1)
    # y_hat1 = lwlr_all(x_mat[100:199], x_mat[0:99], y_mat[0:99], k=0.1)
    # y_test01 = error(y_mat[100:199],y_hat1)
    # y_hat1 = lwlr_all(x_mat[0:99], x_mat[0:99], y_mat[0:99], k=1)
    # y_train1 = error(y_mat[0:99], y_hat1)
    # y_hat1 = lwlr_all(x_mat[100:199], x_mat[0:99], y_mat[0:99], k=1)
    # y_test1 = error(y_mat[100:199], y_hat1)
    # y_hat1 = lwlr_all(x_mat[0:99], x_mat[0:99], y_mat[0:99], k=10)
    # y_train10 = error(y_mat[0:99], y_hat1)
    # y_hat1 = lwlr_all(x_mat[100:199], x_mat[0:99], y_mat[0:99], k=10)
    # y_test10 = error(y_mat[100:199],y_hat1)
    # print(y_train01,y_test01)
    # print(y_train1,y_test1)
    # print(y_train10,y_test10)
    #
    # y_hat = compute_w(x_mat[100:199],x_mat[0:99],y_mat[0:99])
    # print(error(y_mat[100:199],y_hat))





