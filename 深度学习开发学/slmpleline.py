coding="utf-8"
#生成人工数据集
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

np.random.seed(5)#设置随机数种子

x_data=np.linspace(-1,1,100)
y_data=2*x_data+1.0+np.random.randn(*x_data.shape)*0.4
#y=2x+1+噪声

np.random.randn(10)

x_data.shape#x_data.shape的值为一个元组

np.random.randn(*x_data.shape)#拆包，单个表示将元组拆成一个个单独的实参
##np.random.randn(100)#与上式功能相同

y_data=2*x_data+1.0+np.random.randn(100)*0.4
#print(y_data)
#plt.scatter(x_data,y_data)
#plt.show()
#plt.plot(x_data,2*x_data+1.0,color="red",linewidth=3)

#构建模型
x=tf.placeholder("float",name="x")#特征值
y=tf.placeholder("float",name="y")#标签纸

def model(x,w,b):
    return tf.multiply(x,w)+b
w=tf.Variable(1.0,name="w0")#构建现行函数的斜率
b=tf.Variable(0.0,name="b0")#构建截距

pred=model(x,w,b)#pred是预测值，前向计算

#设置训练参数
#迭代次数
train_epochs=10
#学习率
learning_rate=0.05
#控制显示loss值得粒度
display_step=10

#定音损失函数
#采用均方差作为损失函数
loss_function=tf.reduce_mean(tf.square(y-pred))
#定义梯度下降优化器
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)
#创建会话
sess=tf.Session()
#变量初始化
init=tf.global_variables_initializer()
sess.run(init)

step=0#记录训练步数
loss_list=[]#用于保存loss值得列表
#开始训练，，轮数为epoch，采用SGD随机梯度下降优化方法，每轮迭代后，绘出模型曲线
for epoch in range(train_epochs):
    for xs,ys in zip(x_data,y_data):#zip组合成为一个一维数组
        _,loss=sess.run([optimizer,loss_function],feed_dict={x:xs,y:ys})
        #显示损失值
        #display——step控制报告得粒度
        #例如，如果display_step设为2，每两个样本输出一个损失值
        #与超参数不同，该参数不会改变模型得学习规律
        loss_list.append(loss)
        step=step+1
        if step % display_step==0:
            print("train epoch",'%02d'%(epoch+1),"step:%03d"%(step),\
                "loss=","{:.9f}".format(loss))
    b0temp=b.eval(session=sess)
    w0temp=w.eval(session=sess)
    #plt.scatter(x_data,y_data)
    plt.plot(x_data,w0temp*x_data+b0temp)
#添加图示
plt.scatter(x_data,y_data,label="Original data")
plt.plot(x_data,x_data*sess.run(w),\
    label='Fitted  line',color="r",linewidth=3)
plt.plot(loss_list,'g2')
plt.legend(loc=2)#通过参数loc指定图例位置 


plt.show()
print("w:",sess.run(w))
print("b:",sess.run(b))

#利用模型，进行预测
x_test=3.21
predict=sess.run(pred,feed_dict={x:x_test})
print("预测值：%f"%predict)

target=2*x_test+1.0
print("目标值：%f"%target)