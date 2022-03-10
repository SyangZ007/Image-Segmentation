import tensorflow as tf
from tensorflow.keras import layers,Model,Sequential
from tensorflow.keras.activations import sigmoid,softmax
from RSU_Net import REBNCONV,_upsample_like,RSU7,RSU6,RSU5,RSU4,RSU4F,DecodeBlock,SEBlock,AGBlock,AttentionBlock

##############################RSU-Net###########################
class RSUNET1(Model):
    '''一阶剪枝模型'''
    def __init__(self,in_ch=3,out_ch=4,is_muti_output=False):
        super(RSUNET1,self).__init__()
        self.is_muti_output=is_muti_output#是否采用多输出版本模型
        self.stage1 = RSU7(in_ch,32,64)#size:b*h*w*64
        self.pool12 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage2 = RSU6(64,32,128)
        self.pool23 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage3 = RSU5(128,64,256)
        self.pool34 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage4 = RSU4(256,64,256)
        self.pool45 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage5 = RSU4F(256,128,256)
        self.pool56 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage6 = RSU4F(256,128,512)
        self.side6 = layers.Conv2D(4,3,1,padding='same')#,activation='relu')
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
        return tf.math.sigmoid(hx6_side)
#############二阶剪枝模型#################
class RSUNET2(Model):
    '''二阶剪枝模型'''
    def __init__(self,in_ch=3,out_ch=4,is_muti_output=False):
        super(RSUNET2,self).__init__()
        self.is_muti_output=is_muti_output#是否采用多输出版本模型
        self.stage1 = RSU7(in_ch,32,64)#size:b*h*w*64
        self.pool12 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage2 = RSU6(64,32,128)
        self.pool23 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage3 = RSU5(128,64,256)
        self.pool34 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage4 = RSU4(256,64,256)
        self.pool45 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage5 = RSU4F(256,128,256)
        self.pool56 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage6 = RSU4F(256,128,512)
        # attention block
        self.attention1 = AttentionBlock(input_channel=256,mid_channel=128)
        # decoder 五个上采样块 transposed_conv + conv_block   
        self.stage5d = DecodeBlock(128)
        # decoder 6个旁路输出分支,旁路输出模式：3*3卷积输出四通道特征图(线性激活)==>上采样到target size==>所有旁路输出拼接、融合==>1*1卷积、四通道sigmoid激活输出
        self.side5 = layers.Conv2D(4,3,1,padding='same')#,activation='relu')
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
        hx5 = self.attention1(hx5,hx6)#经attention处理后的skip connection
        hx5d = self.stage5d(hx6,hx5)#解码stage1：b*8*50*c
        hx5_side = self.side5(hx5d)#解码网络尺度5旁路输出
        hx5_side = _upsample_like(hx5_side,hx1,mode='normal')
        return tf.math.sigmoid(hx5_side)
#############三阶剪枝模型#################
class RSUNET3(Model):
    '''三阶剪枝模型'''
    def __init__(self,in_ch=3,out_ch=4,is_muti_output=False):
        super(RSUNET3,self).__init__()
        self.is_muti_output=is_muti_output#是否采用多输出版本模型
        self.stage1 = RSU7(in_ch,32,64)#size:b*h*w*64
        self.pool12 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage2 = RSU6(64,32,128)
        self.pool23 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage3 = RSU5(128,64,256)
        self.pool34 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage4 = RSU4(256,64,256)
        self.pool45 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage5 = RSU4F(256,128,256)
        self.pool56 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage6 = RSU4F(256,128,512)
        # attention block
        self.attention1 = AttentionBlock(input_channel=256,mid_channel=128)
        self.attention2 = AttentionBlock(input_channel=256,mid_channel=64)
        # decoder 五个上采样块 transposed_conv + conv_block   
        self.stage5d = DecodeBlock(128)
        self.stage4d = DecodeBlock(64)
        # decoder 6个旁路输出分支,旁路输出模式：3*3卷积输出四通道特征图(线性激活)==>上采样到target size==>所有旁路输出拼接、融合==>1*1卷积、四通道sigmoid激活输出
        self.side4 = layers.Conv2D(4,3,1,padding='same')#,activation='relu')
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
        hx5 = self.attention1(hx5,hx6)#经attention处理后的skip connection
        hx5d = self.stage5d(hx6,hx5)#解码stage1：b*8*50*c
        hx4 = self.attention2(hx4,hx5d)#经attention处理后的skip connection
        hx4d = self.stage4d(hx5d,hx4)#解码stage2：b*16*100*c
        hx4_side = self.side4(hx4d)#解码网络尺度4旁路输出
        hx4_side = _upsample_like(hx4_side,hx1,mode='normal')
        return tf.math.sigmoid(hx4_side)
#############四阶剪枝模型#################
class RSUNET4(Model):
    '''四阶剪枝模型'''
    def __init__(self,in_ch=3,out_ch=4,is_muti_output=False):
        super(RSUNET4,self).__init__()
        self.is_muti_output=is_muti_output#是否采用多输出版本模型
        self.stage1 = RSU7(in_ch,32,64)#size:b*h*w*64
        self.pool12 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage2 = RSU6(64,32,128)
        self.pool23 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage3 = RSU5(128,64,256)
        self.pool34 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage4 = RSU4(256,64,256)
        self.pool45 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage5 = RSU4F(256,128,256)
        self.pool56 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage6 = RSU4F(256,128,512)
        # attention block
        self.attention1 = AttentionBlock(input_channel=256,mid_channel=128)
        self.attention2 = AttentionBlock(input_channel=256,mid_channel=64)
        self.attention3 = AttentionBlock(input_channel=256,mid_channel=64)
        # decoder 五个上采样块 transposed_conv + conv_block   
        self.stage5d = DecodeBlock(128)
        self.stage4d = DecodeBlock(64)
        self.stage3d = DecodeBlock(64)
        # decoder 6个旁路输出分支,旁路输出模式：3*3卷积输出四通道特征图(线性激活)==>上采样到target size==>所有旁路输出拼接、融合==>1*1卷积、四通道sigmoid激活输出
        self.side3 = layers.Conv2D(4,3,1,padding='same')#,activation='relu')
    def call(self,x):
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
        hx5 = self.attention1(hx5,hx6)#经attention处理后的skip connection
        
        hx5d = self.stage5d(hx6,hx5)#解码stage1：b*8*50*c
        hx4 = self.attention2(hx4,hx5d)#经attention处理后的skip connection
        hx4d = self.stage4d(hx5d,hx4)#解码stage2：b*16*100*c
        hx3 = self.attention3(hx3,hx4d)#经attention处理后的skip connection
        hx3d = self.stage3d(hx4d,hx3)#解码stage3：b*32*200*c
        hx3_side = self.side3(hx3d)#解码网络尺度3旁路输出
        hx3_side = _upsample_like(hx3_side,hx1,mode='normal')
        return tf.math.sigmoid(hx3_side)
#############五阶剪枝模型#################
class RSUNET5(Model):
    '''五阶剪枝模型'''
    def __init__(self,in_ch=3,out_ch=4,is_muti_output=False):
        super(RSUNET5,self).__init__()
        self.is_muti_output=is_muti_output#是否采用多输出版本模型
        self.stage1 = RSU7(in_ch,32,64)#size:b*h*w*64
        self.pool12 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage2 = RSU6(64,32,128)
        self.pool23 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage3 = RSU5(128,64,256)
        self.pool34 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage4 = RSU4(256,64,256)
        self.pool45 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage5 = RSU4F(256,128,256)
        self.pool56 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage6 = RSU4F(256,128,512)
        # attention block
        self.attention1 = AttentionBlock(input_channel=256,mid_channel=128)
        self.attention2 = AttentionBlock(input_channel=256,mid_channel=64)
        self.attention3 = AttentionBlock(input_channel=256,mid_channel=64)
        self.attention4 = AttentionBlock(input_channel=128,mid_channel=64)
        # decoder 五个上采样块 transposed_conv + conv_block   
        self.stage5d = DecodeBlock(128)
        self.stage4d = DecodeBlock(64)
        self.stage3d = DecodeBlock(64)
        self.stage2d = DecodeBlock(64)
        # decoder 6个旁路输出分支,旁路输出模式：3*3卷积输出四通道特征图(线性激活)==>上采样到target size==>所有旁路输出拼接、融合==>1*1卷积、四通道sigmoid激活输出
        self.side2 = layers.Conv2D(4,3,1,padding='same')#,activation='relu')
    def call(self,x):
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
        hx5 = self.attention1(hx5,hx6)#经attention处理后的skip connection
        
        hx5d = self.stage5d(hx6,hx5)#解码stage1：b*8*50*c
        hx4 = self.attention2(hx4,hx5d)#经attention处理后的skip connection
        hx4d = self.stage4d(hx5d,hx4)#解码stage2：b*16*100*c
        hx3 = self.attention3(hx3,hx4d)#经attention处理后的skip connection
        hx3d = self.stage3d(hx4d,hx3)#解码stage3：b*32*200*c
        hx2 = self.attention4(hx2,hx3d)#经attention处理后的skip connection
        hx2d = self.stage2d(hx3d,hx2)#解码stage4：b*64*400*c
        hx2_side = self.side2(hx2d)#解码网络尺度2旁路输出
        hx2_side = _upsample_like(hx2_side,hx1,mode='normal')
        return tf.math.sigmoid(hx2_side)
#############六阶剪枝模型#################
class RSUNET6(Model):
    '''六阶剪枝模型'''
    def __init__(self,in_ch=3,out_ch=4,is_muti_output=False):
        super(RSUNET6,self).__init__()
        self.is_muti_output=is_muti_output#是否采用多输出版本模型
        self.stage1 = RSU7(in_ch,32,64)#size:b*h*w*64
        self.pool12 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage2 = RSU6(64,32,128)
        self.pool23 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage3 = RSU5(128,64,256)
        self.pool34 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage4 = RSU4(256,64,256)
        self.pool45 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage5 = RSU4F(256,128,256)
        self.pool56 = layers.MaxPool2D(pool_size=2,strides=2,padding='same')#ceil_mode=True
        self.stage6 = RSU4F(256,128,512)
        # attention block
        self.attention1 = AttentionBlock(input_channel=256,mid_channel=128)
        self.attention2 = AttentionBlock(input_channel=256,mid_channel=64)
        self.attention3 = AttentionBlock(input_channel=256,mid_channel=64)
        self.attention4 = AttentionBlock(input_channel=128,mid_channel=64)
        self.attention5 = AttentionBlock(input_channel=64,mid_channel=64)
        # decoder 五个上采样块 transposed_conv + conv_block   
        self.stage5d = DecodeBlock(128)
        self.stage4d = DecodeBlock(64)
        self.stage3d = DecodeBlock(64)
        self.stage2d = DecodeBlock(64)
        self.stage1d = DecodeBlock(64)
        # decoder 6个旁路输出分支,旁路输出模式：3*3卷积输出四通道特征图(线性激活)==>上采样到target size==>所有旁路输出拼接、融合==>1*1卷积、四通道sigmoid激活输出
        self.side1 = layers.Conv2D(4,3,1,padding='same')#,activation='relu')
    def call(self,x):
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
        hx5 = self.attention1(hx5,hx6)#经attention处理后的skip connection
        
        hx5d = self.stage5d(hx6,hx5)#解码stage1：b*8*50*c
        hx4 = self.attention2(hx4,hx5d)#经attention处理后的skip connection
        hx4d = self.stage4d(hx5d,hx4)#解码stage2：b*16*100*c
        hx3 = self.attention3(hx3,hx4d)#经attention处理后的skip connection
        hx3d = self.stage3d(hx4d,hx3)#解码stage3：b*32*200*c
        hx2 = self.attention4(hx2,hx3d)#经attention处理后的skip connection
        hx2d = self.stage2d(hx3d,hx2)#解码stage4：b*64*400*c

        hx1 = self.attention5(hx1,hx2d)#经attention处理后的skip connection
        hx1d = self.stage1d(hx2d,hx1)#解码stage5：b*128*800*c
        #解码网络尺度1无需旁路输出
        hx1_side = self.side1(hx1d)
        return tf.math.sigmoid(hx1_side)
    
#创建RSU-NET
model1=RSUNET1(in_ch=3,out_ch=4)#num_class=4
model2=RSUNET2(in_ch=3,out_ch=4)#num_class=4
model3=RSUNET3(in_ch=3,out_ch=4)#num_class=4
model4=RSUNET4(in_ch=3,out_ch=4)#num_class=4
model5=RSUNET5(in_ch=3,out_ch=4)#num_class=4
model6=RSUNET6(in_ch=3,out_ch=4)#num_class=4
