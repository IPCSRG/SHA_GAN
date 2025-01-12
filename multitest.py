import argparse
import  time
import cv2,os
import numpy as np
import tensorflow as tf
import neuralgym as ng
from glob import glob
from inpaint_model import InpaintGCModel
from inpaint_ops import random_bbox, bbox2mask, free_form_mask_tf

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_dir', default=r'', type=str,                                                                    #待修复的图像文件名
                    help='The filename of image to be completed.')

parser.add_argument('--checkpoint_dir', default=r'', type=str,
                    help='The directory of tensorflow checkpoint.')

parser.add_argument('--store_dir', default=r'', type=str,                                                                    #待修复的图像文件名
                    help='The filename of image to be completed.')

if __name__ == "__main__":
    # ng.get_gpus(-1)
    args = parser.parse_args()
    file_list_image = glob('{}/*.*'.format(args.dataset_dir))
    save_dir =r''.format(args.store_dir)   # 保存位置
    save_dir_mask = r''.format(args.store_dir)
    save_dir_ori = r''.format(args.store_dir)
    save_dir_comp = r''.format(args.store_dir)
    save_dir_inc = r''.format(args.store_dir)


    for i in range(1):
        dir = str(time.time())
        dir = dir[11:]
        dir_test = save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists( save_dir_mask):
            os.makedirs( save_dir_mask)
        if not os.path.exists(save_dir_ori):
            os.makedirs(save_dir_ori)
        if not os.path.exists(save_dir_comp):
            os.makedirs(save_dir_comp)
        if not os.path.exists(save_dir_inc):
            os.makedirs(save_dir_inc)

        model = InpaintGCModel()
        num = 0
        for j in file_list_image:
            tf.reset_default_graph()
            file_name_im = os.path.basename(j)
            image = cv2.imread(j)
            image = cv2.resize(image,(256,256))
            h, w, _ = image.shape
            grid = 8
            image = image[:h//grid*grid, :w//grid*grid, :]
            print('Shape of image: {}'.format(image.shape))
            image = np.expand_dims(image, 0)
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            with tf.Session(config=sess_config) as sess:
                image = tf.constant(image, dtype=tf.float32)
                k = i//2
                input_image = image
                mask, inc, output, pos = model.build_server_graph(input_image)
                vizx = [mask, inc, output,pos]
                output = tf.concat(vizx, axis=2)
                output = (output + 1.) * 127.5
                output = tf.reverse(output, [-1])
                output = tf.saturate_cast(output, tf.uint8)
                vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                assign_ops = []
                for var in vars_list:
                    vname = var.name
                    from_name = vname
                    var_value = tf.contrib.framework.load_variable(args.checkpoint_dir,from_name)
                    assign_ops.append(tf.assign(var, var_value))
                sess.run(assign_ops)
                print('Model loaded.')
                result = sess.run(output)
                # cv2.imwrite(os.path.join(dir_test,str(num).zfill(5)+'.png'), result[0][:, :, ::-1])
                # cv2.imwrite(os.path.join(save_dir_mask, str(num).zfill(5) + '.png'), result[0][:, :256, ::-1])
                # cv2.imwrite(os.path.join(save_dir_inc, str(num).zfill(5) + '.png'), result[0][:, 256:512, ::-1])
                # cv2.imwrite(os.path.join(save_dir_comp, str(num).zfill(5) + '.png'), result[0][:, 512:768, ::-1])
                # cv2.imwrite(os.path.join(save_dir_ori, str(num).zfill(5) + '.png'), result[0][:, 768:, ::-1])
                # num += 1


                cv2.imwrite(os.path.join(dir_test,file_name_im+'.png'), result[0][:, :, ::-1])
                cv2.imwrite(os.path.join(save_dir_mask, file_name_im+ '.png'), result[0][:, :256, ::-1])
                cv2.imwrite(os.path.join(save_dir_inc, file_name_im+'.png'), result[0][:, 256:512, ::-1])
                cv2.imwrite(os.path.join(save_dir_comp, file_name_im+ '.png'), result[0][:, 512:768, ::-1])
                #cv2.imwrite(os.path.join(save_dir_comp_ori, file_name_im+ '.png'), result[0][:, 768:1024, ::-1])
                cv2.imwrite(os.path.join(save_dir_ori, file_name_im+ '.png'), result[0][:, 768:, ::-1])

                num += 1
                print(num)