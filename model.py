'''
This code refers to https://github.com/xuebinqin/U-2-Net
'''
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

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):
    '''用tf.resize实现，双线性插值上采样到目标尺寸大小,src:待上采样的feats层，tar:上采样后的尺寸与tar层相同'''
    #src = F.upsample(src,size=tar.shape[2:],mode='bilinear')
    src=tf.image.resize(src, size=tar.shape[1:3], method='bilinear')#双线性插值上采样到目标尺寸大小

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

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dilate_rate=1)#输入RSU块第一个卷积块，通道数不变
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

##### 创建U^2-Net模型 ####
class U2NET(Model):
    '''
        标准U^2-Net模型也是U-Net结构
        encoder：RSU7>>>RSU6>>>RSU5>>>RSU4>>>RSU4F>>>Bottle neck(RSU4F)
        decoder：Bottle neck(RSU4F)>>>RSU4F>>>RSU4>>>RSU5>>>RSU6>>>RSU7
        decoder每层feature map均(经过卷积)输出一个mask,
        summary: U2NET总的输出mask，是decoder各尺度输出mask的叠加
        模型返回6个尺度的mask,以及最终的mask d0, 用于计算loss
    '''
    def __init__(self,in_ch=3,out_ch=1):
        super(U2NET,self).__init__()

        self.stage1 = RSU7(in_ch,32,64)#输入通道数3，中间通道数32，输出通道64
        self.pool12 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True

        self.stage2 = RSU6(64,32,128)#输入通道数64，中间通道数32，输出通道128
        self.pool23 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True

        self.stage3 = RSU5(128,64,256)#输入通道数128，中间通道数64，输出通道256
        self.pool34 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True

        self.stage4 = RSU4(256,128,512)#输入通道数256，中间通道数128，输出通道512
        self.pool45 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True

        self.stage5 = RSU4F(512,256,512)#输入通道数512，中间通道数256，输出通道512
        self.pool56 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True

        self.stage6 = RSU4F(512,256,512)#Bottle neck部分

        # decoder 因为U-Net skip connection结构，decoder的输入通道数都扩大两倍
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = layers.Conv2D(out_ch,kernel_size=3,padding='same')#经过一次卷积后，从decoder各尺度分别输出mask
        self.side2 = layers.Conv2D(out_ch,kernel_size=3,padding='same')#经过一次卷积后，从decoder各尺度分别输出mask
        self.side3 = layers.Conv2D(out_ch,kernel_size=3,padding='same')#经过一次卷积后，从decoder各尺度分别输出mask
        self.side4 = layers.Conv2D(out_ch,kernel_size=3,padding='same')#经过一次卷积后，从decoder各尺度分别输出mask
        self.side5 = layers.Conv2D(out_ch,kernel_size=3,padding='same')#经过一次卷积后，从decoder各尺度分别输出mask
        self.side6 = layers.Conv2D(out_ch,kernel_size=3,padding='same')#经过一次卷积后，从decoder各尺度分别输出mask
        #生成mask的卷积，采用sigmoid激活函数
        self.outconv = layers.Conv2D(out_ch,kernel_size=1)#1*1卷积，padding='valid'

    def call(self,x):
        #输入图像x，比如（256，256，3）
        #-------------------- encoder --------------------
        #stage 1
        hx1 = self.stage1(x)
        hx = self.pool12(hx1)
        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)
        #-------------------- decoder --------------------
        hx5d = self.stage5d(tf.concat([hx6up,hx5],-1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(tf.concat([hx5dup,hx4],-1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(tf.concat([hx4dup,hx3],-1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(tf.concat([hx3dup,hx2],-1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(tf.concat([hx2dup,hx1],-1))

        #decoder各尺度side output分别输出mask
        d1 = self.side1(hx1d)#第一个尺度的mask>>>标准size
        
        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)
        ###############将六个尺度的mask concat后，经过一次卷积通道调整####################
        d0 = self.outconv(tf.concat([d1,d2,d3,d4,d5,d6],-1))
        #返回6个尺度的mask,以及最终的mask d0,用于计算loss
        return tf.stack([sigmoid(d0), sigmoid(d1), sigmoid(d2), sigmoid(d3), sigmoid(d4), sigmoid(d5), sigmoid(d6)])

### U^2-Net small ###
class U2NET_Mini(Model):
    '''mini U2-Net'''
    def __init__(self,in_ch=3,out_ch=1,mode='muti_class'):
        super(U2NET_Mini,self).__init__()

        self.mode   = mode
        self.stage1 = RSU7(in_ch,16,64)
        self.pool12 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True

        self.stage2 = RSU6(64,16,64)
        self.pool23 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True

        self.stage3 = RSU5(64,16,64)
        self.pool34 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True

        self.stage4 = RSU4(64,16,64)
        self.pool45 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True

        self.stage5 = RSU4F(64,16,64)
        self.pool56 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True

        self.stage6 = RSU4F(64,16,64)

        # decoder
        self.stage5d = RSU4F(128,16,64)
        self.stage4d = RSU4(128,16,64)
        self.stage3d = RSU5(128,16,64)
        self.stage2d = RSU6(128,16,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = layers.Conv2D(out_ch,kernel_size=3,padding='same')
        self.side2 = layers.Conv2D(out_ch,kernel_size=3,padding='same')
        self.side3 = layers.Conv2D(out_ch,kernel_size=3,padding='same')
        self.side4 = layers.Conv2D(out_ch,kernel_size=3,padding='same')
        self.side5 = layers.Conv2D(out_ch,kernel_size=3,padding='same')
        self.side6 = layers.Conv2D(out_ch,kernel_size=3,padding='same')

        self.outconv = layers.Conv2D(out_ch,kernel_size=1)#1*1卷积，padding='valid'

    def call(self,x):

        #stage 1
        hx1 = self.stage1(x)
        hx = self.pool12(hx1)
        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)
#-------------------- decoder --------------------
        hx5d = self.stage5d(tf.concat([hx6up,hx5],-1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(tf.concat([hx5dup,hx4],-1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(tf.concat([hx4dup,hx3],-1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(tf.concat([hx3dup,hx2],-1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(tf.concat([hx2dup,hx1],-1))

        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(tf.concat([d1,d2,d3,d4,d5,d6],-1))

        if self.mode=='muti_class':
            #多类别语义分割
            return tf.stack([softmax(d0), softmax(d1), softmax(d2), softmax(d3), softmax(d4), softmax(d5), softmax(d6)])
        
        return tf.stack([sigmoid(d0), sigmoid(d1), sigmoid(d2), sigmoid(d3), sigmoid(d4), sigmoid(d5), sigmoid(d6)])

print('successfully build model!')

#################standard U2-Net model########################
#Total params: 43,625,677
#Trainable params: 43,596,877
#Non-trainable params: 28,800
# model=U2NET()
# x=tf.random.normal((1,256,256,3))
# out=model(x)

# for out_ in out:
#     print(out_.shape) 

# model.summary()

#################standard U2-Net model########################
#model_mini=U2NET_Mini()
#x=tf.random.normal((1,320,320,3))
#out=model_mini(x)

#for out_ in out:
    #print(out_.shape) 

#model_mini.summary()
# Total params: 1,115,597
# Trainable params: 1,109,901
# Non-trainable params: 5,696

if __name__=='main':
    model=U2NET()
    x=tf.random.normal((1,320,320,3))
    out=model(x)
    print('successfully build model!')
    print(model.summary())
