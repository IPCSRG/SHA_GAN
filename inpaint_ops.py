import logging
import math
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
import tensorflow.contrib.slim as slim
from neuralgym.ops.layers import resize
from neuralgym.ops.layers import *
from neuralgym.ops.loss_ops import *
from neuralgym.ops.summary_ops import *
from sn import spectral_normed_weight
from vgg19 import Vgg19
# import ops
from tensorflow.keras import layers

logger = logging.getLogger()
np.random.seed(2018)
def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(5, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)

        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)

        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)

        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask

def free_form_mask_tf(parts, maxVertex=16, maxLength=60, maxBrushWidth=14, maxAngle=360, im_size=(256, 256), name='fmask'):
    # mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    with tf.variable_scope(name):
        mask = tf.Variable(tf.zeros([1, im_size[0], im_size[1], 1]), name='free_mask')
        maxVertex = tf.constant(maxVertex, dtype=tf.int32)
        maxLength = tf.constant(maxLength, dtype=tf.int32)
        maxBrushWidth = tf.constant(maxBrushWidth, dtype=tf.int32)
        maxAngle = tf.constant(maxAngle, dtype=tf.int32)
        h = tf.constant(im_size[0], dtype=tf.int32)
        w = tf.constant(im_size[1], dtype=tf.int32)
        for i in range(parts):
            p = tf.py_func(np_free_form_mask, [maxVertex, maxLength, maxBrushWidth, maxAngle, h, w], tf.float32)
            p = tf.reshape(p, [1, im_size[0], im_size[1], 1])
            mask = mask + p
        mask = tf.minimum(mask, 1.0)
    return mask

kernel=tf.constant([
[
[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],
[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],
[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]
],
[
[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],
[[8.,0.,0.],[0.,8.,0.],[0.,0.,8.]],
[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]
],
[
[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],
[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],
[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]
]
])

# 边缘损失
def edge_loss(image,ground_truth ):

    image=(image+1.)*127.5
    E1 = tf.nn.conv2d(image , kernel, [1, 1, 1, 1], padding='SAME',name="e1")
    E2 = tf.nn.conv2d(ground_truth , kernel, [1, 1, 1, 1], padding='SAME',name="e2")
    loss = L1_loss(E2, E1)

    return loss

# gram 矩阵
def get_gram(x):                                                       # def get_gram(x):
    ba,hi,wi,ch = [i.value for i in x.get_shape()]                          # ba,hi,wi,ch=[i.value for i in x.get_shape()]
    feature = tf.reshape(x,[ba,int(hi*wi),ch])                              # feature = tf.reshape(x,[ba,int(hi*wi),ch])
    feature_T = tf.transpose(feature,[0,2,1])                               # feature = tf.transpose(feature,[0,2,1])
    gram = tf.matmul(feature_T,feature)                                     # gram = tf.matmul(feature_T,feature)
    size = 1/(hi*wi*ch)                                                     # size = 1/(hi*wi*ch)
    return gram*size                                                        # return gram*size

# 风格损失
def style_loss(real, fake):
    vgg = Vgg19('vgg19.npy')

    vgg.build(real)

    real_feature_map = vgg.pool1
    real_feature_map1 = vgg.pool2
    real_feature_map2 = vgg.pool3

    vgg.build(fake)
    fake_feature_map = vgg.pool1
    fake_feature_map1 = vgg.pool2
    fake_feature_map2 = vgg.pool3

    gram_comp = get_gram(fake_feature_map)
    gram_gt = get_gram(real_feature_map)

    gram_comp1 = get_gram(fake_feature_map1)
    gram_gt1 = get_gram(real_feature_map1)

    gram_comp2 = get_gram(fake_feature_map2)
    gram_gt2 = get_gram(real_feature_map2)

    loss = L1_loss(gram_comp,gram_gt)+L1_loss(gram_comp1,gram_gt1)+L1_loss(gram_comp2,gram_gt2)

    return loss

def vgg_loss(real, fake):                                     # vgg 损失
    vgg = Vgg19('vgg19.npy')

    vgg.build(real)                                           # vgg.build(real)
    real_feature_map = vgg.conv1_2_no_activation              # real_feature_map = vgg.conv4_4_no_activation
    real_feature_map1 = vgg.conv2_2_no_activation
    real_feature_map2 = vgg.conv3_4_no_activation
    # real_feature_map = vgg.pool1
    # real_feature_map1 = vgg.pool2
    # real_feature_map2 = vgg.pool3

    vgg.build(fake)                                           # vgg.build(fake)
    # fake_feature_map = vgg.pool1
    # fake_feature_map1 = vgg.pool2
    # fake_feature_map2 = vgg.pool3
    fake_feature_map = vgg.conv1_2_no_activation
    fake_feature_map1 = vgg.conv2_2_no_activation
    fake_feature_map2 = vgg.conv3_4_no_activation

    loss = tf.reduce_mean(tf.abs(real_feature_map - fake_feature_map))\
        + 0.5*tf.reduce_mean(tf.abs(real_feature_map1 - fake_feature_map1)) \
        + 0.3*tf.reduce_mean(tf.abs(real_feature_map2 - fake_feature_map2))

    return loss


# VGG 损失
def vgg_loss_(real, fake,bbox):                                     # vgg 损失
    vgg = Vgg19('vgg19.npy')

    vgg.build(real)                                           # vgg.build(real)
    # real_feature_map = vgg.conv4_4_no_activation              # real_feature_map = vgg.conv4_4_no_activation
    real_feature_map = vgg.pool1
    real_feature_map1 = vgg.pool2
    real_feature_map2 = vgg.pool3

    vgg.build(fake)                                           # vgg.build(fake)
    fake_feature_map = vgg.pool1
    fake_feature_map1 = vgg.pool2
    fake_feature_map2 = vgg.pool3

    loss = tf.reduce_mean(tf.abs(local_patch(real_feature_map, (bbox[0] // 2,bbox[1]//2,bbox[2]//2,bbox[3]//2 )) - local_patch(fake_feature_map,(bbox[0] // 2,bbox[1]//2,bbox[2]//2,bbox[3]//2 )) * spatial_discounting_maskx(64,64,0.85)))\
        + tf.reduce_mean(tf.abs(local_patch(real_feature_map1, (bbox[0] // 4,bbox[1]//4,bbox[2]//4,bbox[3]//4 )) - local_patch(fake_feature_map1,(bbox[0] // 4,bbox[1]//4,bbox[2]//4,bbox[3]//4 )) * spatial_discounting_maskx(32,32,0.80))) \
        + tf.reduce_mean(tf.abs(local_patch(real_feature_map2, (bbox[0] // 8,bbox[1]//8,bbox[2]//8,bbox[3]//8 )) - local_patch(fake_feature_map2,(bbox[0] // 8,bbox[1]//8,bbox[2]//8,bbox[3]//8 )) * spatial_discounting_maskx(16,16,0.75)))
    # loss =  L1_loss(real_feature_map1, fake_feature_map1)
    return loss

# L1 重建损失
def L1_loss(x, y): # L1 损失 --> 相减 ， 绝对值 ， 平均
    loss = tf.reduce_mean(tf.abs(x - y)) # loss = tf.reduce_mean(tf.abs(x-y))

    return loss


@add_arg_scope
def gen_conv(x, cnum, ksize, stride=1, rate=1, name='conv',                                                             #生成器卷积
             padding='SAME', activation=tf.nn.relu, training=True):
    """Define conv for generator.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        Rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    x = tf.layers.conv2d(
        x, cnum, ksize, stride, dilation_rate=rate,
        activation=activation, padding=padding, name=name)
    return x


@add_arg_scope
def attention(x, cnum, ksize=1, stride=1, rate=1, name='attention',                                                             #生成器卷积
             padding='SAME', activation=None, training=True):                                                      # 注意力 x[n, 16, 16, 512] ch 512/8 -- 64


    f = gen_conv(x, cnum // 8, ksize, stride, rate, name=name+'_conv_f',  # 生成器卷积
                 padding=padding, activation=activation, training=True)
    g = gen_conv(x, cnum // 8, ksize, stride, rate, name=name+'_conv_g',  # 生成器卷积
                 padding=padding, activation=activation, training=True)
    h = gen_conv(x, cnum , ksize, stride, rate, name=name + '_conv_h',  # 生成器卷积
                 padding=padding, activation=activation, training=True)

            # N = h * w
    s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]                             # [n, 256, 64] x [n, 256, 64]~T -->[n,256,256]

    beta = tf.nn.softmax(s)                                                                                     # attention map

    o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]                                                         # [n,256,256]x[n,256,512] --> [n,256,512]
    gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

    o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]                                                        # --> [n, 16, 16, 512]
    x = gamma * o + x                                                                                           # output

    return x

def hw_flatten(x):                                                                                                     # hw-flatten
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])                                                           # 在保持批轴(轴0)与通道轴（轴3）的同时，使输入张量变平。

def down_sample(x, scale_factor=2) :
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h // scale_factor, w // scale_factor]

    return tf.image.resize_nearest_neighbor(x, size=new_size)

def down_samplex(x, scale) :
    # _, h, w, _ = x.get_shape().as_list()
    # new_size = [h // scale_factor, w // scale_factor]

    return tf.image.resize_nearest_neighbor(x, [scale,scale])

def c_concat(x, y) :
    # _, h, w, _ = x.get_shape().as_list()
    # new_size = [h // scale_factor, w // scale_factor]

    return tf.concat([x,y],axis=-1 )


def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out

def Channel_Attention(x):
    """
    通道注意力
    :param x: 输入数组[b, w, h, filters]
    :return: 输出数组[b, w, h, filters]
    """
    gamma = tf.Variable(tf.ones(1), name='Channel_gamma')
    # gamma = tf.get_variable('sigma_ratio', [], initializer=tf.constant_initializer(0.0))
    x_origin = x
    batch_size, H, W, Channel = x.shape
    x = tf.transpose(x, [0, 3, 1, 2])
    x1 = tf.reshape(x, (-1, Channel, H*W))
    x1_1 = tf.transpose(x1, [0, 2, 1])
    x = tf.matmul(x1, x1_1)
    x = layers.Activation('softmax')(x)
    x = tf.matmul(x, x1)
    x = tf.reshape(x, (-1, Channel, H, W))
    x = tf.transpose(x, [0, 2, 3, 1])
    x = layers.Add()([x*gamma, x_origin])

    return x

@add_arg_scope
def sn_conv1x1(input_, output_dim, update_collection,
              init=tf.contrib.layers.xavier_initializer(), name='sn_conv1x1', activation=tf.nn.relu):
  with tf.variable_scope(name):
    k_h = 1
    k_w = 1
    d_h = 1
    d_w = 1
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=init)
    w_bar = ops.spectral_normed_weight(w, num_iters=1, update_collection=update_collection)

    conv = tf.nn.conv2d(input_, w_bar, strides=[1, d_h, d_w, 1], padding='SAME')
    return conv

@add_arg_scope
def Squeeze_excitation_layer(input_x, ratio=0.5, name='SEnet', training=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        batch_size, hidden_num = input_x.get_shape().as_list()[0], input_x.get_shape().as_list()[3]

        avgpool_channel = tf.reduce_mean(tf.reduce_mean(input_x, axis=1, keepdims=True), axis=2, keepdims=True)
        avgpool_channel = tf.layers.Flatten()(avgpool_channel)

        mlp_1 = tf.layers.dense(avgpool_channel, units=int(hidden_num * ratio), name="mlp_1",
                                    reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        mlp_2 = tf.layers.dense(mlp_1, units=hidden_num, name="mlp_2", reuse=tf.AUTO_REUSE)
        mlp_2 = tf.reshape(mlp_2, [batch_size, 1, 1, hidden_num])
        channel_attention = tf.nn.sigmoid(mlp_2)

        scale = input_x * channel_attention

        return scale

@add_arg_scope
def gen_deconv_fusion(x, cnum, scale=2, name='upsample', padding='SAME', training=True):                                                #生成器反卷积
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.

    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    with tf.variable_scope(name):
        _, h, w, _ = x.get_shape().as_list()
        new_size = [h * scale, w * scale]
        x = tf.image.resize_nearest_neighbor(x, size=new_size)
        x = gen_conv(
            x, cnum, 1, 1, name=name+'_conv', padding=padding,
            training=training)
    return x

@add_arg_scope
def gen_deconv(x, cnum, name='upsample', padding='SAME', training=True):                                                #生成器反卷积
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.

    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    with tf.variable_scope(name):
        x = resize(x, func=tf.image.resize_nearest_neighbor)
        x = gen_conv(
            x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training)
    return x

@add_arg_scope
def gen_deconvx(x, cnum, scale, name='upsample', padding='SAME', training=True):                                                #生成器反卷积
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.

    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    with tf.variable_scope(name):
        x = tf.image.resize_images(x, [scale, scale], method=1)
        # x = resize(x, func=tf.image.resize_nearest_neighbor)
        x = gen_conv(
            x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training)
    return x


@add_arg_scope
def gen_snconv(x, cnum, ksize, stride=1, rate=1, name='conv',  use_bias=True,                                                         #谱正则化卷积
             padding='SAME', activation=tf.nn.relu, training=True):
    """Define spectral normalization conv for discriminator.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        Rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'

    fan_in = ksize * ksize * x.get_shape().as_list()[-1]
    fan_out = ksize * ksize * cnum
    stddev = np.sqrt(2. / (fan_in))
    # initializer for w used for spectral normalization
    w = tf.get_variable(name+"_w", [ksize, ksize, x.get_shape()[-1], cnum],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
    x = tf.nn.conv2d(x, spectral_normed_weight(w, update_collection=tf.GraphKeys.UPDATE_OPS, name=name+"_sn_w"),
                          strides=[1, stride, stride, 1], dilations=[1, rate, rate, 1], padding=padding, name=name)
    if use_bias:
        bias = tf.get_variable(name + "bias", cnum, initializer=tf.constant_initializer(0.0))
        x = tf.nn.bias_add(x, bias)

    return lrelu(x)


@add_arg_scope
def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


@add_arg_scope
def relu(x):
    return tf.nn.relu(x)


@add_arg_scope
def tanh(x):
    return tf.tanh(x)


@add_arg_scope
def gen_deconv(x, cnum, name='upsample', padding='SAME', training=True):                                                #生成器反卷积
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.

    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    with tf.variable_scope(name):
        x = resize(x, func=tf.image.resize_nearest_neighbor)
        x = gen_conv(
            x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training)
    return x

@add_arg_scope
def dis_conv(x, cnum, ksize=5, stride=2, name='conv', training=True):                                                   #辨别器反卷积
    """Define conv for discriminator.
    Activation is set to leaky_relu.

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    x = tf.layers.conv2d(x, cnum, ksize, stride, 'SAME', name=name)
    x = tf.nn.leaky_relu(x)
    return x

@add_arg_scope
def gated_conv(x, cnum, ksize, stride=1, rate=1, name='gated_conv',                                                     #门限卷积
     padding='SAME', activation=tf.nn.relu, training=True):
    """Define gated conv for generator. Add a gating filter

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        Rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    xin = x
    x = tf.layers.conv2d(
        xin, cnum, ksize, stride, dilation_rate=rate,
        activation=activation, padding=padding, name=name)

    gated_mask = tf.layers.conv2d(
        xin, cnum, ksize, stride, dilation_rate=rate,
        activation=tf.nn.sigmoid, padding=padding, name=name+"_mask")

    return x * gated_mask

@add_arg_scope
def eff_gated_conv(x, cnum, ksize, stride=1, rate=1, name='gated_conv',                                                     #门限卷积+
     padding='SAME', activation=tf.nn.relu, training=True):
    """Define gated conv for generator. Add a gating filter

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        Rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    xin = x
    x = tf.layers.conv2d(
        xin, cnum, ksize, stride, dilation_rate=rate,
        activation=activation, padding=padding, name=name)
    xin = tf.nn.avg_pool(xin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding, name=name)
    gated_mask = tf.layers.conv2d(
        xin, cnum, ksize, stride, dilation_rate=rate,
        activation=tf.nn.sigmoid, padding=padding, name=name+"_mask")
    gated_mask = resize(gated_mask, func=tf.image.resize_nearest_neighbor)

    return x * gated_mask

@add_arg_scope
def eff_gated_deconv(x, cnum, ksize, stride=1, rate=1, name='gated_conv',                                                     #门限卷积+
     padding='SAME', activation=tf.nn.relu, training=True):
    """Define gated conv for generator. Add a gating filter

    Args:
        x: Input.
        cnum: Channel number.
        ksize: Kernel size.
        Stride: Convolution stride.
        Rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate*(ksize-1)/2)
        x = tf.pad(x, [[0,0], [p, p], [p, p], [0,0]], mode=padding)
        padding = 'VALID'
    xin = x
    x = resize(x, func=tf.image.resize_nearest_neighbor)
    x = tf.layers.conv2d(
        x, cnum, ksize, stride, dilation_rate=rate,
        activation=activation, padding=padding, name=name)

    gated_mask = tf.layers.conv2d(
        xin, cnum, ksize, stride, dilation_rate=rate,
        activation=tf.nn.sigmoid, padding=padding, name=name+"_mask")
    gated_mask = resize(gated_mask, func=tf.image.resize_nearest_neighbor)

    return x * gated_mask

@add_arg_scope
def gated_deconv(x, cnum, name='upsample', padding='SAME', training=True):                                              #门限反卷积
    """Define gated deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.

    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.

    Returns:
        tf.Tensor: output

    """
    with tf.variable_scope(name):
        x = resize(x, func=tf.image.resize_nearest_neighbor)
        x = gated_conv(
            x, cnum, 3, 1, name=name+'_conv', padding=padding,
            training=training)
    return x


def random_ff_mask(config, name="ff_mask"):                                                                             #随机自由形态掩码
    """Generate a random free form mask with configuration.

    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

    Returns:
        tuple: (top, left, height, width)

    """
    img_shape = config.IMG_SHAPES
    h,w,c = img_shape
    def npmask():

        mask = np.zeros((h,w))
        num_v = 8+np.random.randint(config.MAXVERTEX)                                                                   #tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)

        for i in range(num_v):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            for j in range(1+np.random.randint(5)):
                angle = 0.01+np.random.randint(config.MAXANGLE)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 20+np.random.randint(config.MAXLENGTH)
                brush_w = 25+np.random.randint(config.MAXBRUSHWIDTH)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
        return mask.reshape(mask.shape+(1,)).astype(np.float32)
    with tf.variable_scope(name), tf.device('/cpu:0'):
        mask = tf.py_func(
            npmask,
            [],
            tf.float32, stateful=False)
        mask.set_shape([1,] + [h, w] + [1,])
    return mask

def mask_from_bbox_voc(config, bboxes):
    """
    Use the data from voc dataset. And generate mask from bounding segmentation data

    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

        files: Filename list used for generate bboxes
    Returns:
        a batch of masks
    """
    img_shape = config.IMG_SHAPES
    img_height = img_shape[0]
    img_width = img_shape[1]
    with tf.variable_scope(name), tf.device('/cpu:0'):
        for i in range(files.shapes[0]):
            pass




def mask_from_seg_voc(config, files):
    """
    Use the data from voc dataset. And generate mask from bounding box data

    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

        files: Filename list used for generate bboxes
    Returns:
        a batch of masks
    """
    pass
def random_bbox(config):                                                                                                #随机bbox
    """Generate a random tlhw with configuration.

    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

    Returns:
        tuple: (top, left, height, width)

    """
    img_shape = config.IMG_SHAPES
    img_height = img_shape[0]
    img_width = img_shape[1]
    maxt = img_height - config.VERTICAL_MARGIN - config.HEIGHT
    maxl = img_width - config.HORIZONTAL_MARGIN - config.WIDTH
    t = tf.random_uniform(
        [], minval=config.VERTICAL_MARGIN, maxval=maxt, dtype=tf.int32)
    l = tf.random_uniform(
        [], minval=config.HORIZONTAL_MARGIN, maxval=maxl, dtype=tf.int32)
    h = tf.constant(config.HEIGHT)
    w = tf.constant(config.WIDTH)
    return (t, l, h, w)


def bbox2mask(bbox, config, name='mask'):                                                                               #由bbox生成掩码
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including IMG_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """
    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((1, height, width, 1), np.float32)
        h = np.random.randint(delta_h//2+1)
        w = np.random.randint(delta_w//2+1)
        mask[:, bbox[0]+h:bbox[0]+bbox[2]-h,
             bbox[1]+w:bbox[1]+bbox[3]-w, :] = 1.
        return mask
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img_shape = config.IMG_SHAPES
        height = img_shape[0]
        width = img_shape[1]
        mask = tf.py_func(
            npmask,
            [bbox, height, width,
             config.MAX_DELTA_HEIGHT, config.MAX_DELTA_WIDTH],
            tf.float32, stateful=False)
        mask.set_shape([1] + [height, width] + [1])
    return mask


def mask_patch(x, mask):                                                                                                #mask_patch(x,mask)
    """Crop local patch according to mask.                                                                              根据mask裁取局部块

    Args:
        x: input
        mask: 0,1 mask have the same size of x

    Returns:
        tf.Tensor: local patch

    """
    return x*mask                                                                                                       #return x*mask
def local_patch(x, bbox):                                                                                               #local_patch(x,bbox):根据bbox裁剪出局部块
    """Crop local patch according to bbox.

    Args:
        x: input
        bbox: (top, left, height, width)

    Returns:
        tf.Tensor: local patch

    """
    x = tf.image.crop_to_bounding_box(x, bbox[0], bbox[1], bbox[2], bbox[3])
    return x


def resize_mask_like(mask, x):                                                                                          #resize mask like(mask,x):将mask大小缩放为x大小
    """Resize mask like shape of x.

    Args:
        mask: Original mask.
        x: To shape of x.

    Returns:
        tf.Tensor: resized mask

    """
    mask_resize = resize(
        mask, to_shape=x.get_shape().as_list()[1:3],
        func=tf.image.resize_nearest_neighbor)
    return mask_resize


def spatial_discounting_mask(config):
    """Generate spatial discounting mask constant.

    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        config: Config should have configuration including HEIGHT, WIDTH,
            DISCOUNTED_MASK.

    Returns:
        tf.Tensor: spatial discounting mask
    """
    gamma = config.SPATIAL_DISCOUNTING_GAMMA
    shape = [1, config.HEIGHT, config.WIDTH, 1]              # 孔长宽
    if config.DISCOUNTED_MASK:                               # True
        logger.info('Use spatial discounting l1 loss.')    #
        mask_values = np.ones((config.HEIGHT, config.WIDTH))
        for i in range(config.HEIGHT):
            for j in range(config.WIDTH):
                mask_values[i, j] = max(
                    gamma**min(i, config.HEIGHT-i),
                    gamma**min(j, config.WIDTH-j))
        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 3)
        mask_values = mask_values
    else:
        mask_values = np.ones(shape)
    return tf.constant(mask_values, dtype=tf.float32, shape=shape)

def spatial_discounting_maskx(HEIGHT,WIDTH,gamma):
    """Generate spatial discounting mask constant.

    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        config: Config should have configuration including HEIGHT, WIDTH,
            DISCOUNTED_MASK.

    Returns:
        tf.Tensor: spatial discounting mask
    """
    gamma = gamma
    shape = [1, HEIGHT, WIDTH, 1]              # 孔长宽
    if True:                               # True
        logger.info('Use spatial discounting l1 loss.')    #
        mask_values = np.ones((HEIGHT, WIDTH))
        for i in range(HEIGHT):
            for j in range(WIDTH):
                mask_values[i, j] = max(
                    gamma**min(i, HEIGHT-i),
                    gamma**min(j, WIDTH-j))
        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 3)
        mask_values = mask_values
    # else:
    #     mask_values = np.ones(shape)
    return tf.constant(mask_values, dtype=tf.float32, shape=shape)

def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True):
    """ Contextual attention layer implementation.

    Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        x: Input feature to match (foreground).
        t: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.

    Returns:
        tf.Tensor: output

    """
    # get shapes
    raw_fs = tf.shape(f)
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    # extract patches from background with stride and rate
    kernel = 2*rate
    raw_w = tf.extract_image_patches(
        b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
    raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    f = resize(f, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    b = resize(b, to_shape=[int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate)], func=tf.image.resize_nearest_neighbor)   # https://github.com/tensorflow/tensorflow/issues/11651
    if mask is not None:
        mask = resize(mask, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.split(f, int_fs[0], axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    w = tf.extract_image_patches(
        b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])
    m = tf.extract_image_patches(
        mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = m[0]
    mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0,1,2], keep_dims=True), 0.), tf.float32)
    w_groups = tf.split(w, int_bs[0], axis=0)
    raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
    y = []
    offsets = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])

        # softmax to match
        yi *=  mm  # mask
        yi = tf.nn.softmax(yi*scale, 3)
        yi *=  mm  # mask

        offset = tf.argmax(yi, axis=3, output_type=tf.int32)
        offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
        y.append(yi)
        offsets.append(offset)
    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    offsets = tf.concat(offsets, axis=0)
    offsets.set_shape(int_bs[:3] + [2])
    # case1: visualize optical flow: minus current position
    h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    offsets = offsets - tf.concat([h_add, w_add], axis=3)
    # to flow image
    flow = flow_to_image_tf(offsets)
    # # case2: visualize which pixels are attended
    # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
    if rate != 1:
        flow = resize(flow, scale=rate, func=tf.image.resize_nearest_neighbor)
    return y, flow


def test_contextual_attention(args):
    """Test contextual attention layer with 3-channel image input
    (instead of n-channel feature).

    """
    import cv2
    import os
    # run on cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    rate = 2
    stride = 1
    grid = rate*stride

    b = cv2.imread(args.imageA)
    b = cv2.resize(b, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    h, w, _ = b.shape
    b = b[:h//grid*grid, :w//grid*grid, :]
    b = np.expand_dims(b, 0)
    logger.info('Size of imageA: {}'.format(b.shape))

    f = cv2.imread(args.imageB)
    h, w, _ = f.shape
    f = f[:h//grid*grid, :w//grid*grid, :]
    f = np.expand_dims(f, 0)
    logger.info('Size of imageB: {}'.format(f.shape))

    with tf.Session() as sess:
        bt = tf.constant(b, dtype=tf.float32)
        ft = tf.constant(f, dtype=tf.float32)

        yt, flow = contextual_attention(
            ft, bt, stride=stride, rate=rate,
            training=False, fuse=False)
        y = sess.run(yt)
        cv2.imwrite(args.imageOut, y[0])


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


COLORWHEEL = make_color_wheel()


def compute_color(u,v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img



def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u/(maxrad + np.finfo(float).eps)
        v = v/(maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def flow_to_image_tf(flow, name='flow_to_image'):
    """Tensorflow ops for computing flow to image.
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(flow_to_image, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
        img = img / 127.5 - 1.
        return img


def highlight_flow(flow):
    """Convert flow into middlebury color code image.
    """
    out = []
    s = flow.shape
    for i in range(flow.shape[0]):
        img = np.ones((s[1], s[2], 3)) * 144.
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h,w]
                vi = v[h,w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))


def highlight_flow_tf(flow, name='flow_to_image'):
    """Tensorflow ops for highlight flow.
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(highlight_flow, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
        img = img / 127.5 - 1.
        return img


def image2edge(image):
    """Convert image to edges.
    """
    out = []
    for i in range(image.shape[0]):
        img = cv2.Laplacian(image[i, :, :, :], cv2.CV_64F, ksize=3, scale=2)
        out.append(img)
    return np.float32(np.uint8(out))

class VOCReader(object):
    def __init__(self, path):
        self.path = path

    def load_seg(self, filename):
        file_path = os.path.join(self.path, "SegmentationClass", filename)
        seg = cv2.imread(file_path)
        return seg
    def seg2mask(self, seg):
        pass
    def load_bndbox(self, filename):
        file_path = os.path.join(self.path, "Annotations", filename)
        with open(filename, 'r') as reader:
            xml = reader.read()
        soup = BeautifulSoup(xml, 'xml')
        size = {}
        for tag in soup.size:
            if tag.string != "\n":
                size[tag.name] = int(tag.string)
        objects = soup.find_all('object')
        bndboxs = []
        for obj in objects:
            bndbox = {}
            for tag in obj.bndbox:
                if tag.string != '\n':
                    bndbox[tag.name] = int(tag.string)
            bndboxs.append(bndbox)
        return bndboxs

def blockLayer(x, channels, r, kernel_size=[3,3]):
    output = tf.layers.conv2d(x, channels, (3, 3), padding='same', dilation_rate=(r, r), use_bias=False)
    return tf.nn.relu(output)
   # resdenseblock,
   # 残差密集块  cnum64 layers8 k3 scale1
@add_arg_scope
def resDenseBlock(x, channels=32, layers=8, kernel_size=[3,3], scale=1):
    outputs = [x]
    rates = [2,4,8,16,2,4,8,16]
    for i in range(layers):

        output = blockLayer(tf.concat(outputs[:i],3) if i>=1 else x, channels, rates[i])
        outputs.append(output)

    output = tf.concat(outputs, 3)
    output = slim.conv2d(output, channels, [1,1])
    output *= scale
    return x + output

@add_arg_scope
def upsample(x, scale=2, features=32):                                                 #    x 上采样  scale2   feature64
    output = x
    if (scale & (scale-1)) == 0:                                                       #如果    scale交（scale-1)  ==  0
        for _ in range(int(math.log(scale, 2))):                                 # math.log(100.12) :  4.6063694665635735      math.log(2,2)=1
            output = tf.layers.conv2d(output, 4*features, (3, 3), padding='same', use_bias=False)
            output = pixelshuffle(output, 2)
    elif scale == 3:                                                              #                      如果 scale == 3
        output = tf.layers.conv2d(output, 9*features, (3, 3), padding='same', use_bias=False)       #卷积  ，像素洗牌
        output = pixelshuffle(output, 3)
    else:
        raise NotImplementedError
    return output

def up_sample(x, scale_factor=8) :
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]

    return tf.image.resize_nearest_neighbor(x, size=new_size)

def pixelshuffle(x, upscale_factor):                                                      # shuffle 洗牌 ，侧滑 ， 像素清洗
    return tf.depth_to_space(x, upscale_factor)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageA', default='', type=str, help='Image A as background patches to reconstruct image B.')
    parser.add_argument('--imageB', default='', type=str, help='Image B is reconstructed with image A.')
    parser.add_argument('--imageOut', default='result.png', type=str, help='Image B is reconstructed with image A.')
    args = parser.parse_args()
    test_contextual_attention(args)
