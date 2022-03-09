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
    hx5 = REBNCONV(mid_ch,mid_ch,dilate_rate=2,hx4)#######bottle neck#########
     
    hx4d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1,tf.concat([hx5,hx4],-1))
    hx4dup = _upsample_like(hx4d,hx3)
    hx3d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1,tf.concat([hx4dup,hx3],-1))
    hx3dup = _upsample_like(hx3d,hx2)
    hx2d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1,tf.concat([hx3dup,hx2],-1))
    hx2dup = _upsample_like(hx2d,hx1)
    hx1d = REBNCONV(mid_ch*2,out_ch,dilate_rate=1,tf.concat([hx2dup,hx1],-1))
    return hx1d + hxin

def RSU4(in_ch=3, mid_ch=12, out_ch=3, input_tensor):
    hxin = REBNCONV(in_ch,out_ch,dilate_rate=1,input_tensor)
    hx1 = REBNCONV(out_ch,mid_ch,dilate_rate=1,hxin)
    hx = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(hx1)
    hx2 = REBNCONV(mid_ch,mid_ch,dilate_rate=1,hx)
    hx = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(hx2)
    hx3 = REBNCONV(mid_ch,mid_ch,dilate_rate=1,hx)
    hx4 = REBNCONV(mid_ch,mid_ch,dilate_rate=2,hx3)#######bottle neck#########
     
    hx3d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1,tf.concat([hx4,hx3],-1))
    hx3dup = _upsample_like(hx3d,hx2)
    hx2d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1,tf.concat([hx3dup,hx2],-1))
    hx2dup = _upsample_like(hx2d,hx1)
    hx1d = REBNCONV(mid_ch*2,out_ch,dilate_rate=1,tf.concat([hx2dup,hx1],-1))
    return hx1d + hxin

def RSU4F(in_ch=3, mid_ch=12, out_ch=3, input_tensor):
    hxin = REBNCONV(in_ch,out_ch,dilate_rate=1,input_tensor)
    hx1 = REBNCONV(out_ch,mid_ch,dilate_rate=1,hxin)
    hx2 = REBNCONV(mid_ch,mid_ch,dilate_rate=2,hx1)
    hx3 = REBNCONV(mid_ch,mid_ch,dilate_rate=4,hx2)
    hx4 = REBNCONV(mid_ch,mid_ch,dilate_rate=8,hx3)#######bottle neck#########
     
    hx3d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=4,tf.concat([hx4,hx3],-1))
    hx2d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1,tf.concat([hx3d,hx2],-1))
    hx1d = REBNCONV(mid_ch*2,out_ch,dilate_rate=1,tf.concat([hx2d,hx1],-1))
    return hx1d + hxin

def DecodeBlock(num_filters,input_tensor,skip_tensor):
    x = layers.Conv2DTranspose(filters=num_filters,kernel_size=3,strides=2,padding='same',activation='relu')(input_tensor)#上采样
    x = layers.Conv2D(num_filters/2,kernel_size=3,padding='same',use_bias=False)(tf.concat([x,skip_tensor],axis=-1)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters/2,kernel_size=3,padding='same',use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x                                                                                 

def SEBlock(mid_channel,input_tensor):
    x_avg = layers.GlobalAveragePooling2D()(input_tensor)
    x_avg = layers.Reshape((1,1,mid_channel))(x_avg)   
    x_avg = layers.Conv2D(filters=mid_channel//4,kernel_size=1,padding='same',activation='relu')(x_avg)
    x_avg = layers.Conv2D(filters=mid_channel,kernel_size=1,padding='same',activation='sigmoid')(x_avg)
    out = layers.Multiply()([x_avg,input_tensor])
    return out

def AGBlock(mid_channel,input_tensor,gate):
    phi_g=layers.Conv2D(mid_channel,kernel_size=1,strides=1,padding='same')(gate)
    theta_x=layers.Conv2D(mid_channel,kernel_size=1,strides=2,padding='same')(input_tensor)
    weight_x=layers.Add()([phi_g,theta_x])
    weight_x=layers.Activation('relu')(weight_x)
    weight_x=layers.Conv2D(1,kernel_size=1,strides=1,padding='same',activation='sigmoid)(weight_x)
    weight_x=layers.UpSampling2D(size=(2,2))(weight_x)
    out=layers.Multiply([weight_x,input_tensor])
    return out
                 
def AttentionBlock(input_channel,mid_channel,input_tensor,gate):
    '''[(x==>channel attention)+gate]==>AG Gate(spatial attention)==>AttentionBlock output'''
    x = SEBlock(input_channel,input_tensor)
    out = AGBlock(mid_channel,x,gate)
    return out
##############################RSU-Net###########################
def RSUNET(in_ch=3,out_ch=4,is_muti_output=False,input_tensor):
    '''指定输入tensor，返回RSU-Net Model'''
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
    hx5 = AttentionBlock(256,128,hx5,hx6)
                           
    hx5d = DecodeBlock(128,hx6,hx5)
    hx5_side = layers.Conv2D(4,3,1,padding='same')(hx5d)
    hx5_side = _upsample_like(hx5_side,hx1)
                           
    hx4 = AttentionBlock(256,64,hx4,hx5d)
    hx4d = DecodeBlock(64,hx5d,hx4)                       
    hx4_side = layers.Conv2D(4,3,1,padding='same')(hx4d)
    hx4_side = _upsample_like(hx4_side,hx1)
                          
    hx3 = AttentionBlock(256,64,hx3,hx4d)
    hx3d = DecodeBlock(64,hx4d,hx3)                       
    hx3_side = layers.Conv2D(4,3,1,padding='same')(hx3d)
    hx3_side = _upsample_like(hx3_side,hx1)
                           
    hx2 = AttentionBlock(128,64,hx2,hx3d)
    hx2d = DecodeBlock(64,hx3d,hx2)                       
    hx2_side = layers.Conv2D(4,3,1,padding='same')(hx2d)
    hx2_side = _upsample_like(hx2_side,hx1)
                        
    hx1 = AttentionBlock(64,64,hx1,hx2d)
    hx1d = DecodeBlock(64,hx2d,hx1)                       
    hx1_side = layers.Conv2D(4,3,1,padding='same')(hx1d)
                           
    main_op = layers.Conv2D(out_ch,kernel_size=1,activation='sigmoid')(tf.concat([hx1_side,hx2_side,hx3_side,hx4_side,hx5_side,hx6_side],axis=-1))#主干sigmoid激活输出
    if is_muti_output:
        return Model(inputs=input_tensor,outputs=[main_op,tf.math.sigmoid(hx1_side),tf.math.sigmoid(hx2_side),tf.math.sigmoid(hx3_side),
                           tf.math.sigmoid(hx4_side),tf.math.sigmoid(hx5_side),tf.math.sigmoid(hx6_side)])
    else:
        return Model(inputs=input_tensor,outputs=main_op)

  
  
  
