import tensorflow as tf
from tensorflow.keras import layers,Model,Sequential
from tensorflow.keras.activations import sigmoid,softmax

def REBNCONV(in_ch=3,out_ch=3,dilate_rate=1,input_tensor):
    '''卷积块：Conv+BN+Relu>>>某些层可能会使用空洞卷积'''
    x=layers.Conv2D(out_ch,kernel_size=3,padding='same',use_bias=False,dilation_rate=dilate_rate)(input_tensor)
    x=layers.BatchNormalization()(x)
    out=layers.ReLU()(x)
    return out
def _upsample_like(src,tar,mode='normal'):
    if mode=='normal':
      src=tf.image.resize(src, size=tar.shape[1:3], method='bilinear')#双线性插值上采样到目标尺寸大小
    else:#非normal上采样，采用转置卷积，输出通道数暂定为类别数4
      src=layers.Conv2DTranspose(filters=4,kernel_size=3,strides=2,padding='same',activation='relu')(src)
    return src
 
def RSU7(in_ch=3, mid_ch=12, out_ch=3, input_tensor):
    hxin = REBNCONV(in_ch,out_ch,dilate_rate=1,input_tensor)
    hx1 = REBNCONV(out_ch,mid_ch,dilate_rate=1,hxin)
    hx = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(hx1)
    hx2 = REBNCONV(mid_ch,mid_ch,dilate_rate=1,hx)
    hx = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(hx2)
    hx3 = REBNCONV(mid_ch,mid_ch,dilate_rate=1,hx)
    hx = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(hx3)
    hx4 = REBNCONV(mid_ch,mid_ch,dilate_rate=1,hx)
    hx = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(hx4)
    hx5 = REBNCONV(mid_ch,mid_ch,dilate_rate=1,hx)
    hx = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(hx5)
    hx6 = REBNCONV(mid_ch,mid_ch,dilate_rate=1,hx)
    hx7 = REBNCONV(mid_ch,mid_ch,dilate_rate=2,hx6)#######bottle neck#########
    #######此处为RSU 块的bottle_neck
    hx6d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1,tf.concat([hx7,hx6],-1))
    hx6dup = _upsample_like(hx6d,hx5)
    hx5d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1,tf.concat([hx6dup,hx5],-1))
    hx5dup = _upsample_like(hx5d,hx4)
    hx4d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1,tf.concat([hx5dup,hx4],-1))
    hx4dup = _upsample_like(hx4d,hx3)
    hx3d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1,tf.concat([hx4dup,hx3],-1))
    hx3dup = _upsample_like(hx3d,hx2)
    hx2d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1,tf.concat([hx3dup,hx2],-1))
    hx2dup = _upsample_like(hx2d,hx1)
    hx1d = REBNCONV(mid_ch*2,out_ch,dilate_rate=1,tf.concat([hx2dup,hx1],-1))
    return hx1d + hxin
  
 def RSU6(in_ch=3, mid_ch=12, out_ch=3, input_tensor):
    hxin = REBNCONV(in_ch,out_ch,dilate_rate=1,input_tensor)
    hx1 = REBNCONV(out_ch,mid_ch,dilate_rate=1,hxin)
    hx = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(hx1)
    hx2 = REBNCONV(mid_ch,mid_ch,dilate_rate=1,hx)
    hx = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(hx2)
    hx3 = REBNCONV(mid_ch,mid_ch,dilate_rate=1,hx)
    hx = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(hx3)
    hx4 = REBNCONV(mid_ch,mid_ch,dilate_rate=1,hx)
    hx = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(hx4)
    hx5 = REBNCONV(mid_ch,mid_ch,dilate_rate=1,hx)
    hx6 = REBNCONV(mid_ch,mid_ch,dilate_rate=2,hx5)#######bottle neck#########
 
    hx5d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1,tf.concat([hx6,hx5],-1))
    hx5dup = _upsample_like(hx5d,hx4)
    hx4d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1,tf.concat([hx5dup,hx4],-1))
    hx4dup = _upsample_like(hx4d,hx3)
    hx3d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1,tf.concat([hx4dup,hx3],-1))
    hx3dup = _upsample_like(hx3d,hx2)
    hx2d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1,tf.concat([hx3dup,hx2],-1))
    hx2dup = _upsample_like(hx2d,hx1)
    hx1d = REBNCONV(mid_ch*2,out_ch,dilate_rate=1,tf.concat([hx2dup,hx1],-1))
    return hx1d + hxin
  
 def RSU5(in_ch=3, mid_ch=12, out_ch=3, input_tensor):
    hxin = REBNCONV(in_ch,out_ch,dilate_rate=1,input_tensor)
    hx1 = REBNCONV(out_ch,mid_ch,dilate_rate=1,hxin)
    hx = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(hx1)
    hx2 = REBNCONV(mid_ch,mid_ch,dilate_rate=1,hx)
    hx = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(hx2)
    hx3 = REBNCONV(mid_ch,mid_ch,dilate_rate=1,hx)
    hx = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(hx3)
    hx4 = REBNCONV(mid_ch,mid_ch,dilate_rate=1,hx)
    hx = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(hx4)
    hx5 = REBNCONV(mid_ch,mid_ch,dilate_rate=1,hx)
    hx6 = REBNCONV(mid_ch,mid_ch,dilate_rate=2,hx5)#######bottle neck#########
 
    hx5d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1,tf.concat([hx6,hx5],-1))
    hx5dup = _upsample_like(hx5d,hx4)
    hx4d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1,tf.concat([hx5dup,hx4],-1))
    hx4dup = _upsample_like(hx4d,hx3)
    hx3d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1,tf.concat([hx4dup,hx3],-1))
    hx3dup = _upsample_like(hx3d,hx2)
    hx2d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1,tf.concat([hx3dup,hx2],-1))
    hx2dup = _upsample_like(hx2d,hx1)
    hx1d = REBNCONV(mid_ch*2,out_ch,dilate_rate=1,tf.concat([hx2dup,hx1],-1))
    return hx1d + hxin
  

        self.rebnconvin = REBNCONV(in_ch,out_ch,dilate_rate=1)#输入RSU块第一个卷积块，通道数不变
        #U-nt encoder部分卷积块
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dilate_rate=1)#输入RSU块第二个卷积块，通道数增加
        self.pool1 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        #pooling下采样,pooling stride默认==pooling size,输出尺寸计算采取向上取整ceil
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dilate_rate=1)#输入RSU块第一个卷积块，通道数不变
        self.pool2 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dilate_rate=1)
        self.pool3 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dilate_rate=1)
        #######此处为RSU-5 块的bottle_neck，相比于RSU-7 U-Net encoder少了两层，比RSU-6少了一层
        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dilate_rate=2)#使用了空洞卷积，通道数依旧为mid_ch,
        #######此处为RSU-6 块的bottle_neck
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dilate_rate=1)
    def call(self,x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)#######bottle neck
        hx4d = self.rebnconv4d(tf.concat([hx5,hx4],-1))#skip connection
        hx4dup = _upsample_like(hx4d,hx3)
        hx3d = self.rebnconv3d(tf.concat([hx4dup,hx3],1))#skip connection
        hx3dup = _upsample_like(hx3d,hx2)
        hx2d = self.rebnconv2d(tf.concat([hx3dup,hx2],-1))#skip connection
        hx2dup = _upsample_like(hx2d,hx1)
        hx1d = self.rebnconv1d(tf.concat([hx2dup,hx1],-1))#skip connection
        return hx1d + hxin
  
  
  
  
  
