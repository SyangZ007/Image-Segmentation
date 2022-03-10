import tensorflow as tf
from tensorflow.keras import layers,Model,Sequential
from tensorflow.keras.activations import sigmoid,softmax
from RSU_Net_Functional import RSU7,RSU6,RSU5,RSU4F,_upsample_like,DecodeBlock,AttentionBlock

############结合深度监督机制的剪枝算法#####################
def RSUNET1(in_ch=3,out_ch=4,shape=(128,800,3)):
    '''以第一个旁路分支bottleneck为输出的剪枝模型'''
    input_tensor = tf.keras.Input(shape=shape)
    #stage 1                     
    hx1 = RSU7(in_ch,32,64,input_tensor)
    hx = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(hx1)
    #stage 2
    hx2 = RSU6(64,32,128,hx)
    hx = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(hx2)
    #stage 3
    hx3 = RSU5(128,64,256,hx)
    hx = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(hx3)
    #stage 4
    hx4 = RSU5(256,64,256,hx)
    hx = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(hx4)
    #stage 5
    hx5 = RSU5(256,128,256,hx)
    hx = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(hx5)
    #stage 6 bottle neck
    hx6 = RSU4F(256,128,512,hx)#size:b*4*25*c  相对于原始尺寸32倍下采样
    #decoder解码器部分
    hx6_side = layers.Conv2D(4,3,1,padding='same')(hx6)
    hx6_side = _upsample_like(hx6_side,hx1)
    return Model(inputs=input_tensor,outputs=tf.math.sigmoid(hx6_side))
