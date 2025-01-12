import os
import glob
import socket
import logging
import sys
import tensorflow as tf
import neuralgym as ng
from inpaint_model import InpaintGCModel

logger = logging.getLogger()

def multigpu_graph_def(model, data,  config, gpu_id=0, loss_type='g'):

    files = None
    with tf.device('/cpu:0'):
        if config.MASKFROMFILE:
            images, files = data.data_pipeline(config.BATCH_SIZE)
        else:
            images = data.data_pipeline(config.BATCH_SIZE)


    if gpu_id == 0 and loss_type == 'g':
        _, _, losses = model.build_graph_with_losses(
            images,  config, summary=True, reuse=True)
    else:
        _, _, losses = model.build_graph_with_losses(
            images,  config, reuse=True)
    if loss_type == 'g':
        return losses['g_loss']
    elif loss_type == 'd':
        return losses['d_loss']
    else:
        raise ValueError('loss type is not supported.')


if __name__ == "__main__":
    config = ng.Config('inpaint.yml')
    # if config.GPU_ID != -1:
    #     ng.set_gpus(config.GPU_ID)
    # else:
    #     ng.get_gpus(config.NUM_GPUS)
    # training data
    # Image Data
    with open(config.DATA_FLIST[config.DATASET][0]) as f:
        fnames = f.read().splitlines()
    data = ng.data.DataFromFNames(
        fnames, config.IMG_SHAPES, random_crop=config.RANDOM_CROP)
    images = data.data_pipeline(config.BATCH_SIZE)

    guides = None
    # main model
    model = InpaintGCModel()
    g_vars, d_vars, losses = model.build_graph_with_losses(
        images,   config=config)
    # validation images
    if config.VAL:
        with open(config.DATA_FLIST[config.DATASET][1]) as f:
            val_fnames = f.read().splitlines()
        with open(config.DATA_FLIST[config.MASKDATASET][1]) as f:
            val_mask_fnames = f.read().splitlines()
        # progress monitor by visualizing static images
        for i in range(config.STATIC_VIEW_SIZE):
            static_fnames = val_fnames[i:i+1]
            static_images = ng.data.DataFromFNames(
                static_fnames, config.IMG_SHAPES, nthreads=1,
                random_crop=config.RANDOM_CROP).data_pipeline(1)
            static_masks_fnames =val_mask_fnames[i:i+1]
            static_masks=ng.data.DataFromFNames(
                static_masks_fnames, config.MASK_SHAPES,nthreads=1, random_crop=config.RANDOM_CROP).data_pipeline(1)
            static_inpainted_images = model.build_infer_graph(
                static_images, static_masks,   config, name='static_view/%d' % i)
    # training settings
    lr = tf.get_variable(                                                                                               #lr learning_rate 学习率---   1e-4
        'lr', shape=[], trainable=False,
        initializer=tf.constant_initializer(1e-4))
    d_optimizer = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)                                                      #optimizer
    g_optimizer = d_optimizer
    #AdamOptimizer
    # gradient processor                                                                                                #梯度处理器
    if config.GRADIENT_CLIP:                                                                                            #config.GRADIENT_CLIP: False,,不执行
        gradient_processor = lambda grad_var: (
            tf.clip_by_average_norm(grad_var[0], config.GRADIENT_CLIP_VALUE),
            grad_var[1])
    else:
        gradient_processor = None
    # log dir
    #  #log_dir日志路径
    log_prefix = 'model_logs/' + '_'.join([                                                                             #model_logs/日期机主数据集MASKEDganlogdir
        ng.date_uid(), socket.gethostname(), config.DATASET,
        'MASKED' if config.GAN_WITH_MASK else 'NORMAL',
        config.GAN,config.LOG_DIR])
    # train discriminator with secondary trainer, should initialize before
    # primary trainer.
    # #二级训练器训练辨别器，应先于主训练器进行初始化
    discriminator_training_callback = ng.callbacks.SecondaryTrainer(                                                    #辨别器训练回调
        pstep=1,
        optimizer=d_optimizer,
        var_list=d_vars,
        max_iters=5,
        graph_def=multigpu_graph_def,
        graph_def_kwargs={
            'model': model, 'data': data,   'config': config, 'loss_type': 'd'},
    )
    # train generator with primary trainer                                                                              #主训练器训练生成器
    trainer = ng.train.Trainer(
        optimizer=g_optimizer,
        var_list=g_vars,
        max_iters=config.MAX_ITERS,
        graph_def=multigpu_graph_def,
        grads_summary=config.GRADS_SUMMARY,
        gradient_processor=gradient_processor,
        graph_def_kwargs={
            'model': model, 'data': data,  'config': config, 'loss_type': 'g'},
        spe=config.TRAIN_SPE,
        log_dir=log_prefix,
    )
    # add all callbacks                                                                                                 #添加所有回调
    if not config.PRETRAIN_COARSE_NETWORK:                                                                              #不预训练粗糙网络
        trainer.add_callbacks(discriminator_training_callback)                                                          #训练器添加回调
    trainer.add_callbacks([
        ng.callbacks.WeightsViewer(),                                                                                   #WeightsViewer logs names and size of all weights
        ng.callbacks.ModelRestorer(trainer.context['saver'], dump_prefix='model_logs/'+config.MODEL_RESTORE+'/snap', optimistic=True),
        ng.callbacks.ModelSaver(config.TRAIN_SPE, trainer.context['saver'], log_prefix+'/snap'),
        ng.callbacks.SummaryWriter((config.VAL_PSTEPS//1), trainer.context['summary_writer'], tf.summary.merge_all()),
    ])
    # launch training
    trainer.train()
