#波士盾房价问题


'''第一步：读取数据'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
#读取导入数据文件
df=pd.read_csv("data/boston.csv",header=0)
#输出所有数据
print(df)

#显示数据摘要描述信息,列出统计概述
print(df.describe())

'''第二部：建模准备'''
#数据准备
df=df.values#获取df得值
df=np.array(df)#把df转换为np得数组格式


'''数据归一化'''
for i in range(12):
    df[:,i]=df[:,i]/(df[:,i].max()-df[:,i].min())
'''数据归一化'''


x_data=df[:,:12]#x_data为前12列特征数据
y_data=df[:,12]#y_data为最后一列标签数据
print(y_data,"\n shape=",y_data.shape)
#构建模型
#定义特正数据和标签数据得占位符
x=tf.placeholder(tf.float32,[None,12],name="X")#12个特征数据（12列）None表示数据得行由样本数据决定
y=tf.placeholder(tf.float32,[None,1],name="Y")#一个标签数据（1列

#定义模型函数
#定义了一个，命名空间
with tf.name_scope('Moddel'):
    #w初始化为shape=（12，1）得随机数
    w=tf.Variable(tf.random_normal([12,1],stddev=0.01),name="W")
    #bde 初始化为1.0
    b=tf.Variable(1.0,name="b")

    #w和x是矩阵相乘，用matmul，不能用mutiply或*
    def model(x,w,b):
        return tf.matmul(x,w)+b
    #预测计算操作，前向计算节点
    pred=model(x,w,b)
'''第三步 训练模型'''   
#设置训练参数（超参数）
train_epochs=50#迭代次数
learning_rate=0.01#学习率
#定义均方差损失函数
with tf.name_scope("LossFunction"):
    loss_function=tf.reduce_mean(tf.pow(y-pred,2))#均方误差
#定义梯度下降优化器
'''此时梯度下降具有12元（多个参数元），需要归一化（特征值/（特征值max-min））'''
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)
#创建会话
sess=tf.Session()
#变量初始化
init=tf.global_variables_initializer()
#启动绘画
sess.run(init)

#迭代训练
loss_list=[]#损失值可视化
for epoch in range(train_epochs):
    loss_sum=0.0#损失值和
    for xs,ys in zip(x_data,y_data):
        #feed数据必须和placehoder得shape一致
        xs=xs.reshape(1,12)
        ys=ys.reshape(1,1)
        _,loss=sess.run([optimizer,loss_function],feed_dict={x:xs,y:ys})
        loss_sum=loss_sum+loss
    #打乱数据顺序
    xvalues,yvalues=shuffle(x_data,y_data)

    b0temp=b.eval(session=sess)
    w0temp=w.eval(session=sess)
    loss_average=loss_sum/len(y_data)
    loss_list.append(loss_average)
    print("epoch=",epoch+1,"loss",loss_average,"b=",b0temp,"w=",w0temp)
plt.plot(loss_list)
plt.show()
""" #抽取一条数据进行验证
n=348
x_test=x_data[n]

x_test=x_test.reshape(1,12)
predict=sess.run(pred,feed_dict={x:x_test})
print("预测值：%f"%predict)

target=y_data[n]
print("标签值：%f"%target) """

#可视化代码
sess.close()