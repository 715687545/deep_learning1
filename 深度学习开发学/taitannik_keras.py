#通过高级框架keras预测泰坦尼克号游客生还概率
#第一步，下载相关数据并作处理
import urllib.request
import os 
import pandas as pd
from sklearn import preprocessing#特征值标准化处理
import tensorflow as tf
import matplotlib.pyplot as plt
data_url="http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
data_file_path="data/titanic3.xls"
if not os.path.isfile(data_file_path):
    result=urllib.request.urlretrieve(data_url,data_file_path)
    print("dowmloaded:",result)
else:
    print(data_file_path,'data file already exists.')


#使用pandans处理数据
#读取数据文件，结果为DataFrame格式
df_data = pd.read_excel(data_file_path)
print(type(df_data))
#查看数据再要
#df_data.describe()
selected_cols=['survived','name','pclass','sex','age','sibsp','parch','fare','embarked']
selecter_df_data=df_data[selected_cols]
#print(selecter_df_data)
selecter_df_data.isnull().any()
selecter_df_data.isnull().sum()
#显示存在缺失的行列，确定缺失值的位置
selecter_df_data[selecter_df_data.isnull().values==True]
def prepare_data(df_data):
    df=df_data.drop(['name'],axis=1)#删除姓名列
    age_mean=df['age'].mean()
    df['age']=df['age'].fillna(age_mean)#为缺失age记录填充值
    fare_mean=df['fare'].mean()
    df['fare']=df['fare'].mean()#为缺失的mean值填充均值
    df['sex']=df['sex'].map({'female':0,'male':1}).astype(int)#把性别转化为数值
    df['embarked']=df['embarked'].fillna('S')#补充港口信息
    df['embarked']=df['embarked'].map({'C':0,'Q':1,'S':2}).astype(int)

    ndarry_data =df.values#转化为ndarray数组

    features=ndarry_data[:,1:]#后7列是特征值

    label=ndarry_data[:,0]#第0列是标签值
    #特征值标准化
    minmax_scale=preprocessing.MinMaxScaler(feature_range=(0,1))
    norm_features=minmax_scale.fit_transform(features)

    return norm_features,label
###################
#shuffle 打乱数据顺序，为训练做准备
#通过Pandas的抽样函数sample实现，frac为百分比，原数据集保持不变

shuffled_da_data=selecter_df_data.sample(frac=1)

#数据处理
x_data,y_data=prepare_data(shuffled_da_data)
#划分训练集和测试机
train_size=int(len(x_data)*0.8)
x_train=x_data[:train_size]
y_train=y_data[:train_size]

x_test=x_data[train_size:]
y_test=y_data[train_size:]
##################

#建立Keras序列模型
model=tf.contrib.keras.models.Sequential()

#加入第一层，输入特正数据是7列，也可以用input_shape=(7,)
model.add(tf.contrib.keras.layers.Dense(units=64,
                        input_dim=7,
                        use_bias=True,
                        kernel_initializer='uniform',
                        bias_initializer='zeros',
                        activation='relu'))
model.add(tf.contrib.keras.layers.Dense(units=32,
                        activation='sigmoid'))
model.add(tf.contrib.keras.layers.Dense(units=1,
                        activation='sigmoid'))
model.summary()
#模型设置
model.compile(optimizer=tf.contrib.keras.optimizers.Adam(0.0003),
            loss='binary_crossentropy',
            metrics=['accuracy'])
#模型训练
train_history=model.fit(x=x_train,
                    y=y_train,
                    validation_split=0.2,
                    epochs=100,
                    batch_size=40,
                    verbose=2)
#x:输入特征数据 y:标签数据 验证集比例   verbose:训练过程显示模式 0：不显示，1：带进度条模式2：没epoch显示


print(train_history.history)

#训练过程的历史数据：以字典模式存储
train_history.history.keys()

#训练过程可视化
def visu_train_history(train_history,train_metric,validation_metric):
    plt.plot(train_history.history[train_metric])
    plt.plot(train_history.history[validation_metric])
    plt.title('Train History')
    plt.ylabel(train_metric)
    plt.xlabel('epoch')
    plt.legend(['train','validation'],loc="upper left")
    plt.show()
visu_train_history(train_history,'acc','val_acc')
visu_train_history(train_history,'loss','val_loss')
#模型评估
