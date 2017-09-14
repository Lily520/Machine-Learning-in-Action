# Machine-Learning-in-Action
《机器学习实战》基于python3.6的代码实现

这是我学习《机器学习实战》时的学习笔记和代码实现。

## 下面是每一章的知识总结
### 3.决策树
#### 3.1. 计算以2为底的对数 
```python
    math.log(n,2)
```
#### 3.2. 删除mat或array某一列  
```python
    np.delete(a,index,axis = 1) index可取值[1,2]或3 axis=0／1表示删除行／列
```
#### 3.3. 找到list中出现次数最多的元素 
```python
    from collections import Counter
    most_label = Counter(list).most_common(i) 参数i表示从出现最多的元素开始，取i个
```
#### 3.4. matplotlib注解
    annotate('注释内容’,xy,xytext=None,xycoords=‘data’,textcoords=‘data’,arrowprops=None,**kwargs)
    其中，xy:表示被注释点的位置。xytext:注释文本的坐标位置。 
    arrowprops:表示连接数据点和注释的箭头的类型，该参数为dict类型，该参数有一个名为arrowstyle的键
    xycoords,textcoords:字符串，指示xy,xytext的坐标关系。
    
### 5.Logistci回归
#### 5.1. 构造等差数列  
```python
    np.arange(-4,4,0.1)
```
#### 5.2. 用matplotlib画散点图时，函数scatter()中的参数marker用于设置点的形状
#### 5.3. 用matplotlib画图的模版代码：
```python
        
        import matplotlib.pyplot as plt
        fig = plt.figure()
        Ax = fig.add_subplot(111)
        ax.scatter() #画散点图
        Ax.plot(x,y)
        plt.xlabel("x”)
        plt.ylabel(“y”)
        Plt.show()  
```
#### 5.4. 随机生成指定范围内的整数 
```python
    random.randint(a,b)
```
#### 5.5. 从list中删除指定下表的元素  
```python
    del list[1:3]   // del list[1]
```

### 8.预测数值型数据：回归
#### 8.1 线性回归

##### 8.1.1. 画散点图
```python
        import matplotlib.pyplot as plt
        plt.scatter(x,y,color="r")]
```

#### 8.3 局部加权线性回归

#### 8.3.1. 矩阵在行上合并(行数不变，列数增加)
```python
         np.hstack((x,y))
```

#### 8.3.2. 矩阵在列上合并(行数增加，列数不变)
```python
     np.vstack((x,y))
```
#### 8.3.3. 根据矩阵的某一维进行排序 lambda
```python
     result = sorted(rec.tolist(),key=lambda x:x[0]) #根据矩阵rec的第0列进行排序  注意先将矩阵转为list
```


### 8.4 岭回归

#### 8.4.1. 数据标准化
```python
     from sklearn import preprocessing
     x_norm = preprocessing.scale(x)  #其计算公式为(x-mean(x))/std(x) std表示标准差
```

### 8.5 前向逐步回归

#### 8.5.1. 矩阵乘法
      矩阵对应元素相乘 np.multiply(A,B)
      (2*3)(3*5)的矩阵相乘   np.dot(A,B) 或者 A*B

#### 8.5.2. 数据的标准化  
    将矩阵X标准化成0均值单位方差 (x-np.mean(x,axis=0))/np.var(x,axis=0)   axis=0表示求每一列的方差  axis=1表示求每一行的方差


#### 8.6 乐高玩具套装数据获取 + 交叉验证测试岭回归

#### 8.6.1. 网页数据的分析
     用BeautifulSoup进行


#### 8.6.2. 将列表或array随机打乱,若是多维的，只针对第一维按行打乱 
```python
     np.random.shuffle(a) 
```
#### 8.6.3. 将数据标准化后求出回归系数w。将回归系数w还原
     w_origin = w/var(X)  #X包括测试数据和训练数据，且不含有全为1的一列
     constant = sum(mean(X,0),w_origin) + mean(y) #常数项

#### 8.6.4. 矩阵A的每一项加上常数constant
          A = A + constant
          
          


### 9.树回归
#### 9.1 回归树+模型树+剪枝

##### 9.1.1. 求相关系数
```python
    np.corrcoef(y_hat,y,rowvar = 0)
```
#### 9.2 Tkinter/GUI matplotlib

##### 9.2.1. scatter画散点图函数中的参数s,表示散点的大小
```python
    reDraw.a.scatter(trainData[:,0].T.tolist()[0],trainData[:,1].T.tolist()[0], s = 50)
```
##### 9.2.2. Tkinter
    Button()参数 command表示：回调，当按钮被按下时所调用的函数
    Entry.insert()包含两个参数：第一个是插入的位置，第二个是待插入的值
    Entry.delete(0，END)包含两个参数：第一个是开始索引值，第二个是结束索引值
    Grid()的sticky是设置对齐方式的，默认居中显示
    
    
