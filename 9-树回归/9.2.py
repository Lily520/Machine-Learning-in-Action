from tkinter import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import regTrees
import numpy as np

#画模型树，回归树
def reDraw(tolN,tolS):
    reDraw.f.clf() #清空图像
    reDraw.a = reDraw.f.add_subplot(111)

    trainData = regTrees.loadDataSet(".//machinelearninginaction//ch09//sine.txt")
    if chkBtnVar.get(): #复选框被选中，求模型树
        modelTree = regTrees.createTree(trainData,leafType=regTrees.modelLeaf, errType= regTrees.modelErr, ops = (tolS,tolN))
        y_hat = regTrees.createForeCast(modelTree,trainData[:,0],modelEval=regTrees.modelTreeEval)
    else: #回归树
        modelTree = regTrees.createTree(trainData, ops=(tolS, tolN))
        y_hat = regTrees.createForeCast(modelTree, trainData[:, 0])
    data_hat = np.hstack((trainData[:,0],y_hat))
    sort_hat = sorted(data_hat.tolist(),key = lambda x:x[0])
    sort_x = [x[0] for x in sort_hat]
    sort_y = [x[1] for x in sort_hat]
    reDraw.a.scatter(trainData[:,0].T.tolist()[0],trainData[:,1].T.tolist()[0], s = 50)
    reDraw.a.plot(sort_x,sort_y)
    reDraw.canvas.show()

#求tolN,tolS
def getInputs():
    try: tolN = int(tolNentry.get())
    except:
        tolN = 10
        print("tolN is integer!")
    tolNentry.delete(0,END)
    tolNentry.insert(0,tolN)

    try: tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("tolS is float!")
    tolSentry.delete(0,END)
    tolSentry.insert(0,tolS)
    return tolN,tolS

#点击Button:ReDraw 时回调的函数
def drawNewTree():
    tolN,tolS = getInputs()
    reDraw(tolN,tolS)

root = Tk()
#Label
Label(root,text = "Plot Place Holder").grid(row = 0,columnspan = 2)
Label(root,text = "tolN").grid(row = 1)
Label(root,text = "tolS").grid(row = 2)
#Entry
tolNentry = Entry(root)
tolNentry.grid(row = 1,column = 1)
tolNentry.insert(0,10)
tolSentry = Entry(root)
tolSentry.grid(row = 2,column = 1)
tolSentry.insert(0,1.0)
#Button
Button(root,text = "ReDraw",command = drawNewTree).grid(row = 2,column = 2)
#checkButton IntVar
chkBtnVar = IntVar()
chkBtn = Checkbutton(root,text = "Model Tree", variable = chkBtnVar)
chkBtn.grid(row = 3,columnspan = 2)

#在TK的GUI上放置一个画布，并用grid调整布局
reDraw.f = Figure(figsize=(5,4),dpi = 100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f,master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0,columnspan=2)


root.mainloop()
