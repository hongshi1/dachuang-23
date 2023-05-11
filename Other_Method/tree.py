from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt #这个包貌似经常被用到
from math import log # 干嘛的?
import operator
import pickle

def createDataSet():
    dataSet = [[0,0,0,0,'no'],
               [0,0,0,1,'no'],
               [0,1,0,1,'yes'],
               [0,1,1,0,'yes'],
               [0,0,0,0,'no'],
               [1,0,0,1,'no'],
               [1,0,0,0,'no'],
               [1,1,1,1,'yes'],
               [1,0,1,2,'yes'],
               [2,0,1,2,'yes'],
               [2,0,1,1,'yes'],
               [2,1,0,1,'yes'],
               [2,1,0,2,'yes'],
               [2,0,0,0,'no'],
               ]
    labels = ['f1-age','f2-work','f3-home','f4-loan']
    #这个大致上是一个是否愿意借贷款的问题  前面的表示年龄?? f2表示有无稳定哦工作 f3表示有无房屋 f4表示贷款的金额?
    return dataSet,labels
    # 突然发现 ,这个所谓的返回其实可以被理解为一个固定的代码块!
    # 另外 这个是一个二分类的问题 分成yes和no两个类

def createTree(dataset,labels,featLabels): #三个参数分别表示 每次被切分的数据集 对应的标号,以及数选择特征的排序
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):  #后面是所有样本的个数 ,前面是 classlist中 第一个的个数
        return classList[0]  #返回类别
    if len(dataSet[0]) == 1:  # 如果只有一个类别 说明已经遍历完了  ? 为什么还有一个len
        return majorityCnt(classList)  #返回最多的类别

    #选择当前的最优特征
    bestFeat = chooseBestFeatureToSplit(dataset) #假设返回的是索引
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}} #用当前获得的最好特征构造一个树
    #删掉当前的特征 ??这么牛  还能这样子删数据?
    del labels[bestFeat]



