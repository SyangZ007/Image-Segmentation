import tensorflow as tf
from tensorflow.keras import layers,Model,Sequential
from tensorflow.keras.activations import sigmoid,softmax
from RSU_Net import RSU4F,RSU4,RSU5,RSU6,RSU7,AttentionBlock,DecodeBlock
import segmentation_models as sm
############################Metrics#############################
#Metrics
class MeanIOU(tf.keras.metrics.Metric):
    '''计算所有类别的mean iou:等效于sm.base.functional.iou_score(pt,pr,backend=tf.keras.backend,per_image=True)'''
    def __init__(self, name='mean_iou',threshold=0.5,class_weight=1.,**kwargs):
        super(MeanIOU, self).__init__(name=name, **kwargs)
        self.iou = self.add_weight(name='iou', shape=(),initializer='zeros')
        self.threshold=threshold
    def update_state(self, y_true, y_pred, sample_weight=None, weight=[0.3, 0.4, 0.1, 0.2]):
        y_pred=tf.where(y_pred>=self.threshold,1.,0.)#根据阈值输出mask shape;[b,h,w,4]
        #除开batch维度与channel维度，其余各个轴分别计算，也即对每个样本各通道(各通道代表不同类)分别计算IOU值 #shape:[b,4]
        intersection = tf.math.reduce_sum(y_pred*y_true,axis=(1,2))
        union = tf.math.reduce_sum(y_true+y_pred,axis=(1,2)) - intersection#shape:[b,4]
        iou = tf.reduce_mean((intersection + 1e-6) / (union + 1e-6),axis=0)#per_image=True,对batch维度求平均 shape:[1,4]
        self.iou.assign(tf.reduce_mean(iou))#计算每个step的Mean IOU
    def result(self):
        return self.iou
class IOU_PerClass(tf.keras.metrics.Metric):
    '''指定类别index，计算对应index单个类的iou'''
    def __init__(self,name='single_cls_iou',threshold=0.5,index=1,**kwargs):
        super(IOU_PerClass, self).__init__(name=name, **kwargs)
        self.idx=index
        self.threshold=threshold
        self.per_iou = self.add_weight(name='single_iou',shape=(),initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.per_iou.assign(sm.base.functional.iou_score(y_true, y_pred,threshold=self.threshold,class_indexes=[self.idx],
                                         backend=tf.keras.backend,per_image=True))
    def result(self):
        return self.per_iou
#CallBacks
class BatchMeric(tf.keras.callbacks.Callback):#每个batch迭代完记录loss
    def __init__(self):#定义需要记录的指标
        self.total_loss_per_batch=[]
        self.target_loss_per_batch=[]
        self.iou_per_batch=[]
        self.iou1_per_batch=[]#class 1 的iou变化
        self.iou2_per_batch=[]#class 2 的iou变化
        self.iou3_per_batch=[]#class 3 的iou变化
        self.iou4_per_batch=[]#class 4 的iou变化
        
    def on_train_batch_end(self, batch, logs={}):
        self.total_loss_per_batch.append(logs['total loss'])
        self.target_loss_per_batch.append(logs['target loss'])
        self.iou_per_batch.append(logs['mean_iou'])
        self.iou1_per_batch.append(logs['cls1_iou'])
        self.iou2_per_batch.append(logs['cls2_iou'])
        self.iou3_per_batch.append(logs['cls3_iou'])
        self.iou4_per_batch.append(logs['cls4_iou'])

####################training metrics######################
total_loss_metric=tf.keras.metrics.Mean(name="total_loss")#记录total loss metric
target_loss_metric=tf.keras.metrics.Mean(name="target_loss")#记录target loss metric

mean_iou_metric=MeanIOU(name='mean_iou')
mean_iou_1=IOU_PerClass(name='cls1_iou',index=0)
mean_iou_2=IOU_PerClass(name='cls2_iou',index=1)
mean_iou_3=IOU_PerClass(name='cls3_iou',index=2)
mean_iou_4=IOU_PerClass(name='cls4_iou',index=3)
######################loss部分#######################
losses1=sm.losses.DiceLoss(class_weights=[2.5,3.,1.,1.5],per_image=True)
###################################################
losses2=sm.losses.BinaryFocalLoss(alpha=0.25, gamma=2.0)
losses=losses1+losses2
###################3添加两个变量到Graph###############
alpha=tf.Variable(1.,trainable=False)#旁路分支损失权重
STEP=tf.constant(7000,'float32')#总迭代步数
@tf.function
def fusion_loss(d0, d1, d2, d3, d4, d5, d6, y_true):
    tar_loss = losses(y_true,d0)#模型主干输出，采用BCE+DICE损失
    #旁路输出采用DICE损失监督训练
    loss1 = losses(y_true,d1)
    loss2 = losses(y_true,d2)
    loss3 = losses(y_true,d3)
    loss4 = losses(y_true,d4)
    loss5 = losses(y_true,d5)
    loss6 = losses(y_true,d6)
    total_loss = tar_loss + alpha*0.4*(loss1+loss2+loss3+loss4+loss5+loss6)
    alpha.assign_sub(1/STEP)#旁路分支的损失贡献会衰减到0
    return tar_loss,total_loss
def fusion_loss_val(d0, d1, d2, d3, d4, d5, d6, y_true):
    tar_loss = losses(y_true,d0)#模型主干输出，采用BCE+DICE损失
    #旁路输出采用DICE损失监督训练
    loss1 = losses(y_true,d1)
    loss2 = losses(y_true,d2)
    loss3 = losses(y_true,d3)
    loss4 = losses(y_true,d4)
    loss5 = losses(y_true,d5)
    loss6 = losses(y_true,d6)
    total_loss = tar_loss + alpha*0.4*(loss1+loss2+loss3+loss4+loss5+loss6)
    return tar_loss,total_loss
##############################RSU-Net###########################
class RSUNET(Model):
    '''RSUNET根据U-Net结构形式以RSU块搭建，解码部分为普通Decoder,每个Decoder各尺度旁路输出一个mask，
    上采样到统一尺度输出最终mask
    超参数设计：编码[32,64,128,128,256,512]+解码通道数变化[256,128,128,64,32]'''
    def __init__(self,in_ch=3,out_ch=4,is_muti_output=False):
        super(RSUNET,self).__init__()
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
        if self.is_muti_output:
            #多输出版，同时监督训练主干输出与多个旁路分支输出
            return main_op,tf.math.sigmoid(hx1_side),tf.math.sigmoid(hx2_side),tf.math.sigmoid(hx3_side),tf.math.sigmoid(hx4_side),tf.math.sigmoid(hx5_side),tf.math.sigmoid(hx6_side)
        else:
            return main_op#仅监督训练主干输出版本
    def compile(self, optimizer, loss_fn):
        super(RSUNET, self).compile()
        self.optimizer = optimizer#传入优化器
        self.loss_fn = loss_fn#定义损失函数
    def train_step(self, data):
        imgs,masks=data
        # Train themodel
        with tf.GradientTape() as tape:
            d0, d1, d2, d3, d4, d5, d6 = self(imgs,training=True)#1、前向传播
            target_loss,total_loss = self.loss_fn(d0, d1, d2, d3, d4, d5, d6, masks)#2、loss计算及梯度更新参数
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))#
        #3、metrics更新
        total_loss_metric.update_state(total_loss),target_loss_metric.update_state(target_loss)
        mean_iou_metric.update_state(masks,d0),mean_iou_1.update_state(masks,d0)
        mean_iou_2.update_state(masks,d0),mean_iou_3.update_state(masks,d0),mean_iou_4.update_state(masks,d0)
        return {"total loss": total_loss_metric.result(), "target loss": target_loss_metric.result(),
                "mean_iou": mean_iou_metric.result(),"cls1_iou": mean_iou_1.result(),
               "cls2_iou": mean_iou_2.result(),"cls3_iou": mean_iou_3.result(),"cls4_iou": mean_iou_4.result()}
    def test_step(self, data):
        imgs,masks=data
        d0, d1, d2, d3, d4, d5, d6 = self(imgs,training=False)#1、前向传播
        target_loss,total_loss = fusion_loss_val(d0, d1, d2, d3, d4, d5, d6, masks)
        total_loss_metric.update_state(total_loss),target_loss_metric.update_state(target_loss)
        mean_iou_metric.update_state(masks,d0),mean_iou_1.update_state(masks,d0)
        mean_iou_2.update_state(masks,d0),mean_iou_3.update_state(masks,d0),mean_iou_4.update_state(masks,d0)
        return {"total loss": total_loss_metric.result(), "target loss": target_loss_metric.result(),
                "mean_iou": mean_iou_metric.result(),"cls1_iou": mean_iou_1.result(),
               "cls2_iou": mean_iou_2.result(),"cls3_iou": mean_iou_3.result(),"cls4_iou": mean_iou_4.result()}
    @property
    def metrics(self):
        return [total_loss_metric,target_loss_metric, mean_iou_metric,mean_iou_1,mean_iou_2,mean_iou_3,mean_iou_4]
