import tensorflow as tf
from tensorflow.keras import layers,Model,Sequential
from tensorflow.keras.activations import sigmoid,softmax

class REBNCONV(layers.Layer):
    '''卷积块：Conv+BN+Relu>>>某些层可能会使用空洞卷积'''
    def __init__(self,in_ch=3,out_ch=3,dilate_rate=1):
        super(REBNCONV,self).__init__()
        self.conv=layers.Conv2D(out_ch,kernel_size=3,padding='same',
                    use_bias=False,dilation_rate=dilate_rate)
        self.bn=layers.BatchNormalization()
        self.relu=layers.ReLU()
    def call(self,x):
        out = self.relu(self.bn(self.conv(x)))
        return out

def _upsample_like(src,tar,mode='normal'):
    '''用tf.resize实现，双线性插值上采样到目标尺寸大小,src:待上采样的feats层，tar:上采样后的尺寸与tar层相同，
      指定mode以更改上采样方式'''
    if mode=='normal':
      src=tf.image.resize(src, size=tar.shape[1:3], method='bilinear')#双线性插值上采样到目标尺寸大小
    else:#非normal上采样，采用转置卷积，输出通道数暂定为类别数4
      src=layers.Conv2DTranspose(filters=4,kernel_size=3,strides=2,padding='same',activation='relu')(src)
    return src
#RSU块其实是一个U-Net结构，输入及输出feature map尺寸不变
### RSU-7 ###
class RSU7(Model):
    '''RSU7 U-Net块'''
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()
        self.rebnconvin = REBNCONV(in_ch,out_ch,dilate_rate=1)#输入RSU块第一个卷积块，通道数不变
        #U-nt encoder部分卷积块
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dilate_rate=1)#输入RSU块第二个卷积块，通道数增加
        self.pool1 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        #pooling下采样,pooling stride默认==pooling size,输出尺寸计算采取向上取整ceil
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dilate_rate=1)#输入RSU块第3个卷积块，通道数不变
        self.pool2 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dilate_rate=1)
        self.pool3 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dilate_rate=1)
        self.pool4 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')
        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dilate_rate=1)
        self.pool5 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')
        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dilate_rate=1)
        #######此处为RSU 块的bottle_neck
        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dilate_rate=2)#使用了空洞卷积，通道数依旧为mid_ch,
        #######此处为RSU 块的bottle_neck
        #U-nt decoder部分卷积块
        #由于U-Net skip connection输入通道数扩充为了两倍，输出通道数依旧为Mid_ch
        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1)
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
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)#######bottle neck
        hx6d =  self.rebnconv6d(tf.concat([hx7,hx6],-1))#skip connection
        hx6dup = _upsample_like(hx6d,hx5)
        hx5d =  self.rebnconv5d(tf.concat([hx6dup,hx5],-1))#skip connection
        hx5dup = _upsample_like(hx5d,hx4)
        hx4d = self.rebnconv4d(tf.concat([hx5dup,hx4],-1))#skip connection
        hx4dup = _upsample_like(hx4d,hx3)
        hx3d = self.rebnconv3d(tf.concat([hx4dup,hx3],1))#skip connection
        hx3dup = _upsample_like(hx3d,hx2)
        hx2d = self.rebnconv2d(tf.concat([hx3dup,hx2],-1))#skip connection
        hx2dup = _upsample_like(hx2d,hx1)
        hx1d = self.rebnconv1d(tf.concat([hx2dup,hx1],-1))#skip connection
        return hx1d + hxin
### RSU-6 ###
class RSU6(Model):#UNet06DRES(nn.Module):
    '''RSU6 U-Net块,相比RSU7少了一层'''
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()
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
        self.pool4 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')
        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dilate_rate=1)
        #######此处为RSU-6 块的bottle_neck，相比于RSU-7 U-Net encoder少了一层
        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dilate_rate=2)#使用了空洞卷积，通道数依旧为mid_ch,
        #######此处为RSU-6 块的bottle_neck
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=1)
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
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx)#######bottle neck
        hx5d =  self.rebnconv5d(tf.concat([hx6,hx5],-1))#skip connection
        hx5dup = _upsample_like(hx5d,hx4)
        hx4d = self.rebnconv4d(tf.concat([hx5dup,hx4],-1))#skip connection
        hx4dup = _upsample_like(hx4d,hx3)
        hx3d = self.rebnconv3d(tf.concat([hx4dup,hx3],1))#skip connection
        hx3dup = _upsample_like(hx3d,hx2)
        hx2d = self.rebnconv2d(tf.concat([hx3dup,hx2],-1))#skip connection
        hx2dup = _upsample_like(hx2d,hx1)
        hx1d = self.rebnconv1d(tf.concat([hx2dup,hx1],-1))#skip connection
        return hx1d + hxin
### RSU-5 ###
class RSU5(Model):#UNet05DRES(nn.Module):
    '''RSU5 U-Net块,相比RSU6少了一层'''
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()
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
### RSU-4 ###
class RSU4(Model):
    '''RSU4 U-Net块,相比RSU5少了一层'''
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()
        self.rebnconvin = REBNCONV(in_ch,out_ch,dilate_rate=1)
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dilate_rate=1)
        self.pool1 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dilate_rate=1)
        self.pool2 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dilate_rate=1)
        #######此处为RSU-4 块的bottle_neck，比RSU-5少了一层
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dilate_rate=2)
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
        hx4 = self.rebnconv4(hx3)#######bottle neck
        hx3d = self.rebnconv3d(tf.concat([hx4,hx3],1))#skip connection
        hx3dup = _upsample_like(hx3d,hx2)
        hx2d = self.rebnconv2d(tf.concat([hx3dup,hx2],-1))#skip connection
        hx2dup = _upsample_like(hx2d,hx1)
        hx1d = self.rebnconv1d(tf.concat([hx2dup,hx1],-1))#skip connection
        return hx1d + hxin

### RSU-4F ###
class RSU4F(Model):
    '''
    RSU4F U-Net块,较RSU4、5、6、7结构不同，没有采用maxpooling下采样(同样没有上采样)，特征图大小为变化
    encoder：两个普通卷积块+空洞卷积dilate=2+空洞卷积dilate=4>>>bottle neck 空洞卷积dilate=8
    decoder: bottle neck>>>空洞卷积dilate=4+空洞卷积dilate=2+一个普通卷积块
    '''
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()
        self.rebnconvin = REBNCONV(in_ch,out_ch,dilate_rate=1)
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dilate_rate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dilate_rate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dilate_rate=4)
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dilate_rate=8)#bottle_neck
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dilate_rate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dilate_rate=1)
    def call(self,x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)#bottle_neck
        #print(hx4.shape,hx3.shape)
        hx3d = self.rebnconv3d(tf.concat([hx4,hx3],-1))#skip connection
        hx2d = self.rebnconv2d(tf.concat([hx3d,hx2],-1))#skip connection
        hx1d = self.rebnconv1d(tf.concat([hx2d,hx1],-1))#skip connection
        return hx1d + hxin
#上采样块
class DecodeBlock(Model):
    '''ConvBN+ConvBN形式。指定DecodeBlock上采样块的通道数，上采样通道数不变==>卷积块通道数压缩为1/2'''
    def __init__(self, num_filters):
        super(DecodeBlock,self).__init__()
        #两种上采样方式，通过UPSampling2D或者转置卷积实现
        self.upsample=layers.Conv2DTranspose(filters=num_filters,kernel_size=3,strides=2,padding='same',activation='relu')
        self.conv_block=Sequential([layers.Conv2D(num_filters/2,kernel_size=3,padding='same',use_bias=False),
                  layers.BatchNormalization(),layers.Activation('relu'),
                  layers.Conv2D(num_filters/2,kernel_size=3,padding='same',use_bias=False),
                  layers.BatchNormalization(),layers.Activation('relu')])
        self.concat=layers.Concatenate(axis=-1)
    def call(self,x,skip_x):#完成上采样后的DecodeBlock
        x=self.upsample(x)
        return self.conv_block(self.concat([x,skip_x]))
#Attention Block
class AttentionBlock(Model):
    '''
    Arg：gate，decoder待上采样输出 x，encoder skip connection输出 mid_channel，通道数
    Input:x[b,w,h,c1], gate[b,w/2,h/2,c2]
    return：叠加了attention权重的skip connection输出
    '''
    def __init__(self, mid_channel):
        super(AttentionBlock,self).__init__()
        self.conv1=layers.Conv2D(mid_channel,kernel_size=1,strides=1,padding='same')
        self.conv2=layers.Conv2D(mid_channel,kernel_size=1,strides=2,padding='same')
        self.add=layers.Add()
        self.act1=layers.Activation('relu')
        self.conv3=layers.Conv2D(1,kernel_size=1,strides=1,padding='same')
        self.act2=layers.Activation('sigmoid')
        self.upsampling=layers.UpSampling2D(size=(2,2))
        self.multiply=layers.Multiply()
    def call(self,x,gate):
        phi_g=self.conv1(gate)#[b,w/2,h/2,c_mid]
        theta_x=self.conv2(x)#[b,w/2,h/2,c_mid]
        #attention权值矩阵weight_x
        weight_x=self.add([phi_g,theta_x])
        weight_x=self.act1(weight_x)
        weight_x=self.conv3(weight_x)#[b,w/2,h/2,1]
        weight_x=self.act2(weight_x)#sigmoid输出0~1权值矩阵
        weight_x=self.upsampling(weight_x)#[b,w,h,1]
        weight_x=tf.repeat(weight_x,repeats=x.shape[-1],axis=-1)#[b,w,h,c1]
        return self.multiply([weight_x,x])
##############################RSU-Net###########################
class RSUNET(Model):
    '''RSUNET根据U-Net结构形式以RSU块搭建，解码部分为普通Decoder,每个Decoder各尺度旁路输出一个mask，
    上采样到统一尺度输出最终mask
    超参数设计：编码[32,64,128,128,256,512]+解码通道数变化[256,128,128,64,32]'''
    def __init__(self,in_ch=3,out_ch=4):
        super(RSUNET,self).__init__()
        self.stage1 = RSU7(in_ch,16,32)#size:b*h*w*64
        self.pool12 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage2 = RSU6(32,16,64)
        self.pool23 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage3 = RSU5(64,16,64)
        self.pool34 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage4 = RSU4(64,16,128)
        self.pool45 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage5 = RSU4F(128,16,256)
        self.pool56 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage6 = RSU4F(256,16,512)
        # attention block
        self.attention1 = AttentionBlock(256)
        self.attention2 = AttentionBlock(128)
        self.attention3 = AttentionBlock(128)
        self.attention4 = AttentionBlock(64)
        self.attention5 = AttentionBlock(32)
        # decoder 五个上采样块 transposed_conv + conv_block   
        self.stage5d = DecodeBlock(256)
        self.stage4d = DecodeBlock(128)
        self.stage3d = DecodeBlock(128)
        self.stage2d = DecodeBlock(64)
        self.stage1d = DecodeBlock(32)
        # decoder 6个旁路输出分支,旁路输出模式：3*3卷积输出四通道特征图(线性激活)==>上采样到target size==>所有旁路输出拼接、融合==>1*1卷积、四通道sigmoid激活输出
        self.side6 = layers.Conv2D(4,3,1,padding='same')#,activation='relu')
        self.side5 = layers.Conv2D(4,3,1,padding='same')#,activation='relu')
        self.side4 = layers.Conv2D(4,3,1,padding='same')#,activation='relu')
        self.side3 = layers.Conv2D(4,3,1,padding='same')#,activation='relu')
        self.side2 = layers.Conv2D(4,3,1,padding='same')#,activation='relu')
        self.side1 = layers.Conv2D(4,3,1,padding='same')#,activation='relu')
        self.outconv = layers.Conv2D(out_ch,kernel_size=1,activation='sigmoid')#输出层1*1卷积调整通道数，padding='valid'
    def call(self,x):
        #stage 1
        hx1 = self.stage1(x)#size:b*128*800*c
        hx = self.pool12(hx1)#size:b*64*400*c
        #stage 2
        hx2 = self.stage2(hx)#size:b*64*400*c
        hx = self.pool23(hx2)#size:b*32*200*c
        #stage 3
        hx3 = self.stage3(hx)#size:b*32*200*c
        hx = self.pool34(hx3)#size:b*16*100*c
        #stage 4
        hx4 = self.stage4(hx)#size:b*16*100*c
        hx = self.pool45(hx4)#size:b*8*50*c
        #stage 5
        hx5 = self.stage5(hx)#size:b*8*50*c
        hx = self.pool56(hx5)#size:b*4*25*c
        #stage 6 bottle neck
        hx6 = self.stage6(hx)#size:b*4*25*c  相对于原始尺寸32倍下采样
        #decoder解码器部分
        hx6_side = self.side6(hx6)#解码网络各尺度6旁路输出
        hx6_side = _upsample_like(hx6_side,hx1,mode='normal')

        hx5 = self.attention1(hx5,hx6)#经attention处理后的skip connection
        
        hx5d = self.stage5d(hx6,hx5)#解码stage1：b*8*50*c
        hx5_side = self.side5(hx5d)#解码网络尺度5旁路输出
        hx5_side = _upsample_like(hx5_side,hx1,mode='normal')
        
        hx4 = self.attention2(hx4,hx5d)#经attention处理后的skip connection
        hx4d = self.stage4d(hx5d,hx4)#解码stage2：b*16*100*c
        hx4_side = self.side4(hx4d)#解码网络尺度4旁路输出
        hx4_side = _upsample_like(hx4_side,hx1,mode='normal')

        hx3 = self.attention3(hx3,hx4d)#经attention处理后的skip connection
        hx3d = self.stage3d(hx4d,hx3)#解码stage3：b*32*200*c
        hx3_side = self.side3(hx3d)#解码网络尺度3旁路输出
        hx3_side = _upsample_like(hx3_side,hx1,mode='normal')

        hx2 = self.attention4(hx2,hx3d)#经attention处理后的skip connection
        hx2d = self.stage2d(hx3d,hx2)#解码stage4：b*64*400*c
        hx2_side = self.side2(hx2d)#解码网络尺度2旁路输出
        hx2_side = _upsample_like(hx2_side,hx1,mode='normal')

        hx1 = self.attention5(hx1,hx2d)#经attention处理后的skip connection
        hx1d = self.stage1d(hx2d,hx1)#解码stage5：b*128*800*c
        #解码网络尺度1无需旁路输出
        hx1_side = self.side1(hx1d)
        
        main_op=self.outconv(tf.concat([hx1_side,hx2_side,hx3_side,hx4_side,hx5_side,hx6_side],axis=-1))#主干sigmoid激活输出
        #return main_op#仅监督训练主干输出版本
        
        #多输出版，同时监督训练主干输出与多个旁路分支输出
        return main_op,tf.math.sigmoid(hx1_side),tf.math.sigmoid(hx2_side),tf.math.sigmoid(hx3_side),tf.math.sigmoid(hx4_side),tf.math.sigmoid(hx5_side),tf.math.sigmoid(hx6_side)
    #subclass compile
    def compile(self, optimizer, loss_fn):#, metrics):
        super(RSUNET, self).compile()
        self.opt = optimizer
        #self.metrics = metrics
        self.loss_fn = loss_fn#loss_fn计算多输出的损失，返回target_loss,total_loss
    #subclass model.fit train step
    def train_step(self, data):
        #imgs, gt_masks = data
        imgs, gt_masks = next(iter(data))
        with tf.GradientTape() as tape:
            d0, d1, d2, d3, d4, d5, d6 = self(imgs, training=True) # Forward pass
            # Compute our own loss
            tar_loss,total_loss = self.loss_fn(d0, d1, d2, d3, d4, d5, d6, gt_masks)#接受主干输出+6个旁路输出，返回target_loss,total_loss
        # 使用total loss计算gradients
        trainable_vars = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        # Update weights
        self.opt.apply_gradients(zip(grads, trainable_vars))
        # Compute our own metrics
        #self.metrics.update_state(y, d0)
        #return {"mean iou": self.metrics.result()}
    #@property
    #def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        #return [self.metrics]
##############################Auto-Encoder###########################
#Auto-Encoder上采样块
class Auto_DecodeBlock(Model):
    '''Auto_DecodeBlock，无skip connection'''
    def __init__(self, num_filters):
        super(Auto_DecodeBlock,self).__init__()
        self.conv_block=Sequential([layers.Conv2DTranspose(filters=num_filters,kernel_size=3,strides=2,padding='same',activation='relu'),
                  layers.Conv2D(num_filters/2,kernel_size=3,padding='same',use_bias=False),
                  layers.BatchNormalization(),layers.Activation('relu'),
                  layers.Conv2D(num_filters/2,kernel_size=3,padding='same',use_bias=False),
                  layers.BatchNormalization(),layers.Activation('relu')])
    def call(self,x):#完成上采样后的DecodeBlock
        return self.conv_block(x)
class AutoEncoder(Model):
    '''根据RSU-Net的结构搭建AutoEncoder，提取其encoder权重作为RSU-Net训练的预训练权重'''
    def __init__(self,in_ch=3,out_ch=3):
        super(AutoEncoder,self).__init__()
        self.stage1 = RSU7(in_ch,16,32)#size:b*h*w*64
        self.pool12 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage2 = RSU6(32,16,64)
        self.pool23 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage3 = RSU5(64,16,64)
        self.pool34 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage4 = RSU4(64,16,128)
        self.pool45 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage5 = RSU4F(128,16,256)
        self.pool56 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage6 = RSU4F(256,16,512)
        # AutoEncoder不需要attention，skip connection，旁路输出分支
        # decoder 五个上采样块 transposed_conv + conv_block   
        self.stage5d = Auto_DecodeBlock(256)
        self.stage4d = Auto_DecodeBlock(128)
        self.stage3d = Auto_DecodeBlock(128)
        self.stage2d = Auto_DecodeBlock(64)
        self.stage1d = Auto_DecodeBlock(32)
        
        self.outconv = layers.Conv2D(out_ch,kernel_size=1,activation='sigmoid')#输出层1*1卷积调整通道数，padding='valid'
    def call(self,x):
        #stage 1
        hx1 = self.stage1(x)#size:b*128*800*c
        hx = self.pool12(hx1)#size:b*64*400*c
        #stage 2
        hx2 = self.stage2(hx)#size:b*64*400*c
        hx = self.pool23(hx2)#size:b*32*200*c
        #stage 3
        hx3 = self.stage3(hx)#size:b*32*200*c
        hx = self.pool34(hx3)#size:b*16*100*c
        #stage 4
        hx4 = self.stage4(hx)#size:b*16*100*c
        hx = self.pool45(hx4)#size:b*8*50*c
        #stage 5
        hx5 = self.stage5(hx)#size:b*8*50*c
        hx = self.pool56(hx5)#size:b*4*25*c
        #stage 6 bottle neck
        hx6 = self.stage6(hx)#size:b*4*25*c  相对于原始尺寸32倍下采样
        #decoder解码器部分
        hx5d = self.stage5d(hx6)#解码stage1：b*8*50*c
        hx4d = self.stage4d(hx5d)#解码stage2：b*16*100*c
        hx3d = self.stage3d(hx4d)#解码stage3：b*32*200*c
        hx2d = self.stage2d(hx3d)#解码stage4：b*64*400*c
        hx1d = self.stage1d(hx2d)#解码stage5：b*128*800*c
        
        return self.outconv(hx1d)#sigmoid激活输出
#创建RSU-NET
model=RSUNET(in_ch=3,out_ch=4)#num_class=4
auto_encoder=AutoEncoder(in_ch=3,out_ch=3)#输入输出均为三通道图像
######提取AutoEncoder权重初始化RSU-Net
#for l1, l2 in zip(model.layers[:11],auto_encoder.layers[0:11]):
    #l1.set_weights(l2.get_weights())
