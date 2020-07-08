import tensorflow as tf
""" hello = tf.constant([1,2,3])
sess = tf.Session()
try:
    print(sess.run(hello))
except:
    print("eorr")
finally:
    sess.close() """
""" import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(0,20,100)
plt.plot(x,np.sin(x))
plt.show() """
value=tf.Variable(0,name="value")
one=tf.constant(1)
new_value=tf.add(value,one)
update_value=tf.assign(value,new_value)#更新变量值

first=tf.Variable(0,name="first")
answer=tf.add(first,value)
update_value1=tf.assign(first,answer)

init=tf.global_variables_initializer()#初始化所有变量
# with tf.Session() as sess:#
#     sess.run(init)        #
sess=tf.Session()           #
sess.run(init)              #两种创建会话的方式
for _ in range(10):
    sess.run(update_value)
    sess.run(update_value1)
    #print(sess.run(value))
print(sess.run(first))