import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'#cpu加速指令
# -*- coding: utf-8 -*-
#初始化参数与文件目录
#存储模型的粒度
save_step=5
ckpt_dir="./trains_data/ckpt_dir/"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt

mnist=input_data.read_data_sets("./MNIST_data/",one_hot=True)
#print("训练集 train 数量：",mnist.train.num_examples,",验证集 validation 数量：",mnist.validation.num_examples,",测试集 test 数量",mnist.test.num_examples)
#len(mnist.train.images[0])
#mnist.train.images[0].reshape(28,28)#变为28*28的矩阵


#################
#mnist中每种图片像素28*28，定义需要，的占位符
x=tf.placeholder(tf.float32,[None,784],name="X")#像素占位符：784位，行待定
y=tf.placeholder(tf.float32,[None,10],name="Y")#one_hot占位符，10位，行待定


""" #构建隐藏层
H1_NN=256#第一隐藏层神经元为256个
H2_NN=64#第二隐藏层神经元为64个

#输入层--第一层隐藏参数和偏置项
W1=tf.Variable(tf.truncated_normal([784,H1_NN],stddev=0.1))
b1=tf.Variable(tf.zeros([H1_NN]))

#第一隐藏层-第二隐藏层参数和偏置项
W2=tf.Variable(tf.truncated_normal([H1_NN,H2_NN],stddev=0.1))
b2=tf.Variable(tf.zeros([H2_NN]))

#第二隐藏层-输出层参数和偏置项
W3=tf.Variable(tf.truncated_normal([H2_NN,10],stddev=0.1))
b3=tf.Variable(tf.zeros([10]))

#计算第一隐藏层结果
Y1=tf.nn.relu(tf.matmul(x,W1)+b1)

#计算第二隐藏层结果
Y2=tf.nn.relu(tf.matmul(Y1,W2)+b2)

#计算输出结果
forward=tf.matmul(Y2,W3)+b3
pred=tf.nn.softmax(forward)
 """
#模型重构
#定义全连接层函数
def fvn_layer(inputs,
            input_dim,
            output_dim,
            activation=None):##输入数据  输入神经元 输出神经元数量  激活函数
    W=tf.Variable(tf.truncated_normal([input_dim,output_dim],stddev=0.1))
    #以截断正态随机数初始化W
    b=tf.Variable(tf.zeros([output_dim]))
    #以0 初始化b
    XWb=tf.matmul(inputs,W)+b#建立表达式

    if activation is None:#默认不使用激活函数
        outputs=XWb
    else:
        outputs=activation(XWb)
    
    return outputs

h1=fvn_layer(inputs=x,
            input_dim=784,
            output_dim=256,
            activation=tf.nn.relu)

h2=fvn_layer(inputs=h1,
            input_dim=256,
            output_dim=64,
            activation=tf.nn.relu)

h3=fvn_layer(h2,64,32,None)
forward=fvn_layer(h3,32,10,None)
pred=tf.nn.softmax(forward)
#存储模型
saver=tf.train.Saver()
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
    if(epoch+1)%save_step==0:
        saver.save(sess,
        os.path.join(ckpt_dir,'mnist_h256_model_{:06d}.ckpt').format(epoch+1))
    print("Model{:06d} saved!".format(epoch+1))
saver.save(sess,os.path.join(ckpt_dir,'mnist_h256_model.ckpt'))
print("Model saved!")

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
