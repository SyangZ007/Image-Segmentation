# %% [code] {"execution":{"iopub.status.busy":"2021-06-25T08:55:39.832538Z","iopub.execute_input":"2021-06-25T08:55:39.832892Z","iopub.status.idle":"2021-06-25T08:55:46.653200Z","shell.execute_reply.started":"2021-06-25T08:55:39.832814Z","shell.execute_reply":"2021-06-25T08:55:46.652402Z"}}
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import fashion_mnist


# %% [code]
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#归一化
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
print(x_train.shape,x_test.shape)

train_set=tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(128)
test_set=tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)

# %% [code] {"jupyter":{"outputs_hidden":true}}
#创建一个隐藏层数为2的神经网络 
#网络结构：[b,28*28] => [b,256] => [b,128] => [b,10]
#创建tf.Variable类型的网络各训练参数
w1 = tf.Variable(tf.random.normal(shape=(28*28,256),stddev=0.1))#kernel1,(28*28,256)
b1 = tf.Variable(tf.zeros((256)))#bias intialize to 0
w2 = tf.Variable(tf.random.normal(shape=(256,128),stddev=0.1))#kernel
b2 = tf.Variable(tf.zeros((128)))#bias intialize to 0
w3 = tf.Variable(tf.random.normal(shape=(128,10),stddev=0.1))#kernel1
b3 = tf.Variable(tf.zeros((10)))#bias intialize to 0

# %% [markdown]
# **build networks and training**

# %% [code] {"execution":{"iopub.status.busy":"2021-06-25T09:36:56.128303Z","iopub.execute_input":"2021-06-25T09:36:56.128649Z","iopub.status.idle":"2021-06-25T09:37:39.181510Z","shell.execute_reply.started":"2021-06-25T09:36:56.128622Z","shell.execute_reply":"2021-06-25T09:37:39.180583Z"},"jupyter":{"outputs_hidden":true}}
lr=1e-3
epochs=10
loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#@tf.function
def train(epochs=10,lr=1e-3):
    losses=[]
    for epoch in range(epochs):
        for step,(x,y) in enumerate(train_set):
            x = tf.reshape(x,(-1,28*28))#flatten
            with tf.GradientTape() as tape:
                #tf.GradientTape只能对tf.variable类型跟踪求导，否则tape.watch()
                h = tf.nn.relu(x@w1 + b1) #auto broadcast.h:[b,256]
                h = tf.nn.relu(h@w2 + b2) #h:[b,128]
                y_pred = h@w3 + b3 #h:[b,10] 线性输出logits,from_logits=True
                ###compute loss####
                loss = loss_fn(y,y_pred)

            grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])#对指定的变量求导
            ##############反向传播更新tf.variable参数权值################
            #tf.Variable.assign_sub()方法
            w1.assign_sub(lr*grads[0])#equal to:w1=w1 - lr*grads[0]
            b1.assign_sub(lr*grads[1])
            w2.assign_sub(lr*grads[2])
            b2.assign_sub(lr*grads[3])
            w3.assign_sub(lr*grads[4])
            b3.assign_sub(lr*grads[5])

            if step % 10 ==0:
                losses.append(loss)
        
            if step % 100 ==0:
                tf.print('current in epoch:{},loss:{}'.format(epoch,loss))
    return losses

train_loss=train(epochs,lr)

# %% [code] {"execution":{"iopub.status.busy":"2021-06-25T09:38:34.578458Z","iopub.execute_input":"2021-06-25T09:38:34.578779Z","iopub.status.idle":"2021-06-25T09:38:34.957851Z","shell.execute_reply.started":"2021-06-25T09:38:34.578753Z","shell.execute_reply":"2021-06-25T09:38:34.956830Z"}}
plt.plot(train_loss)

