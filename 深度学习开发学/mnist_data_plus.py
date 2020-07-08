import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'#cpu加速指令
# -*- coding: utf-8 -*-


import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt

mnist=input_data.read_data_sets("./MNIST_data/",one_hot=True)
#print("训练集 train 数量：",mnist.train.num_examples,",验证集 validation 数量：",mnist.validation.num_examples,",测试集 test 数量",mnist.test.num_examples)
len(mnist.train.images[0])
mnist.train.images[0].reshape(28,28)#变为28*28的矩阵

#################
#mnist中每种图片像素28*28，定义需要，的占位符
x=tf.placeholder(tf.float32,[None,784],name="X")#像素占位符：784位，行待定
y=tf.placeholder(tf.float32,[None,10],name="Y")#one_hot占位符，10位，行待定


#构建隐藏层
H1_NN=256
W1=tf.Variable(tf.random_normal([784,H1_NN]))
b1=tf.Variable(tf.zeros([H1_NN]))
Y1=tf.nn.relu(tf.matmul(x,W1)+b1)

#构建输出层

W2=tf.Variable(tf.random_normal([H1_NN,10]))
b2=tf.Variable(tf.zeros([10]))

forward=tf.matmul(Y1,W2)+b2
pred=tf.nn.softmax(forward)

#

#定义损失函数（交叉熵损失函数）

###loss_function=tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
#tensorflow提供了softmax_cross_entropy_with_logits函数
#用于避免因为log（0）值位NaN造成的数据不稳定

loss_function=tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=forward,labels=y))
########
#训练模型
####
#设置训练参数
train_epochs=50#训练论述
batch_size=100 #单次训练样本数
total_batch=int(mnist.train.num_examples/batch_size)#一次训练有多少批次
display_step=1#显示粒度
learning_rate=0.01#学习率


#选择梯度下降优化器
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

#定义准确率
#检查预测类别tf.argmax(pred,1)与实际类别tf.argmax(y,1)的匹配情况
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(pred,1))

#准确率将布尔值转变位浮点数，并计算平均值
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#cast，把pre转化位浮点数

#训练模型
from time import time
startTime=time()

sess=tf.Session()#声明会话

init=tf.global_variables_initializer()#全局初始化

sess.run(init)

########
#模型训练
for epoch in range(train_epochs):
    for batch in range(total_batch):
        xs,ys=mnist.train.next_batch(batch_size)#读取批次数据
        sess.run(optimizer,feed_dict={x:xs,y:ys})#执行批次训练

    #total_batch个批次训练集完成后，使用验证数据计算误差和准确率，验证集没有分批
    loss,acc=sess.run([loss_function,accuracy],\
        feed_dict={x:mnist.validation.images,y:mnist.validation.labels})
    #打印训练过程中的信息
    if(epoch+1)%display_step==0:
        print("train epoch:",'%02d'%(epoch+1),"loss=","{:.9f}".format(loss),\
            "Accuracy=","{:.4f}".format(acc))
#显示运行总时间
duration=time()-startTime

print("Train Finished! Which takes:","{:.2f}".format(duration))

####pi评估模型
#在测试集上评估模型准确率
accu_test=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
print("test accuracy:",accu_test)


###模型应用与可视化
#应用若认为准确率可以接受,进行预测
prediction_result=sess.run(tf.argmax(pred,1),feed_dict={x:mnist.test.images})

prediction_result[0:10]

#定义一个输出错误分类的函数
import numpy as np
def print_opredict_errs(labels,prediction):#标签列表，，预测值列表
    count=0
    compare_list=(prediction==np.argmax(labels,1))
    err_lists=[i for i in range(len(compare_list)) if compare_list[i]==False]

    for x in err_lists:
        print("index="+str(x)+"标签值=",np.argmax(labels[x]),"预测值=",prediction[x])
        count=count+1
    print("总计："+str(count))
    return err_lists#返回未匹配正确的列表
test=print_opredict_errs(labels=mnist.test.labels,
prediction=prediction_result)
#test0=tuple(test)
#定义可视化函数
import matplotlib.pyplot as plt
import numpy as np

def plot_image_labels_prediction(images, labels,prediction,*index,num=10,start=2): 
    #图像列表#标签列表#预测值列表。从第index个开始显示#缺省显示数量
    fig=plt.gcf()#获取当前图表
    fig.set_size_inches(10,12)#1英寸等于2.54cm
    if num-start>25:
        print("the number output ont out 25")
        #num=25#限制最多25个例
        num=start+25
    for j in index:
        for i in range(start,num) :
            ax=plt.subplot(5,5,i-start+1)#获取当前要处理的子图
            print(type(index))
        
            k=j[i]
            print(type(j))
            print(k)
            ax.imshow(np.reshape(images[k],(28,28)),
            cmap='binary')

            title="lable="+str(np.argmax(labels[k]))#构架图上要显示的要素

            if len(prediction)>0:
                title+=",predict="+str(prediction[k])

            ax.set_title(title,fontsize=10)#显示图上title信息
            ax.set_xticks([])#不显示坐标轴
            ax.set_yticks([])
        #index+=1
        plt.show()
        break

plot_image_labels_prediction(mnist.test.images,
                            mnist.test.labels,
                            prediction_result, 
                            test,num=50,start=20) 
'''

fig=plt.gcf()#获取当前图表
fig.set_size_inches(10,12)#1英寸等于2.54cm
#if num>25:
num=25#限制最多25个例
for i in range(0,num):
    listarry=test[i]
    ax=plt.subplot(5,5,i+1)#获取当前要处理的子图
    ax.imshow(np.reshape(mnist.test.images[listarry],(28,28)),
        cmap='binary')

    title="lable="+str(np.argmax(mnist.test.labels[listarry]))#构架图上要显示的要素

    if len(prediction_result)>0:
        title+=",predict="+str(prediction_result[listarry])

    ax.set_title(title,fontsize=10)#显示图上title信息
    ax.set_xticks([])#不显示坐标轴
    ax.set_yticks([])
    #index+=1
plt.show()'''