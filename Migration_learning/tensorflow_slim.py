"""
Created on Wed Jun  6 11:56:58 2018

@author: zy
"""
'''
利用已经训练好的vgg16网络对flowers数据集进行微调
把最后一层分类由2000->5 然后重新训练，我们也可以冻结其它所有层，只训练最后一层
'''
from tensorflow.contrib.slim.nets import vgg
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import migration_config as cfg
from xie_pascal_voc_load import pascal_voc_xie
import tensorflow.contrib.slim as slim
from tensorflow.python.tools import inspect_checkpoint as chkp
DATA_DIR = './datasets/data/flowers'
# 输出类别
NUM_CLASSES = 5

# 获取图片大小
IMAGE_SIZE = vgg.vgg_16.default_image_size

def get_tuned_variable(exclusions):
    exclusions_list = [scope.strip() for scope in exclusions]
    variable_to_restore = []
    for var in slim.get_model_variables():
        exclude = False
        for exclusions_index in exclusions_list:
            if var.op.name.startswith(exclusions_index):
                exclude = True
                break
        if not exclude:
            variable_to_restore.append(var)
    return variable_to_restore

def get_trainable_variables(inclusiongs):
    inclusiongs_list = [scope.strip() for scope in inclusiongs]
    variables_to_train = []
    for scope in inclusiongs_list:
        variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variable)
    return variables_to_train





def flowers_fine_tuning():
    '''
    演示一个VGG16的例子
    微调 这里只调整VGG16最后一层全连接层，把1000类改为5类
    对网络进行训练
    '''

    '''
    1.设置参数，并加载数据
    '''
    # 用于保存微调后的检查点文件和日志文件路径
    train_log_dir = './log/vgg16/fine_tune'
    train_log_file = 'flowers_fine_tune.ckpt'

    # 官方下载的检查点文件路径
    checkpoint_file = './vgg_16_2016_08_28/vgg_16.ckpt'
    chkp.print_tensors_in_checkpoint_file(checkpoint_file, tensor_name='', all_tensors=False, all_tensor_names=True)
    #打印出checkpoint_file中tensor的名字！

    # 设置batch_size
    batch_size = cfg.BATCH_SIZE


    # 训练集数据长度
    n_train = 3293
    # 测试集数据长度
    n_test = 377
    # 迭代轮数
    training_epochs = cfg.EPOCH
    display_epoch = 1

    #display_epoch = 1

    if not os.path.isdir(train_log_dir):
        os.makedirs(train_log_dir)
    train = pascal_voc_xie(phase='train')
    test = pascal_voc_xie(phase='test')
    # 加载数据
    # train_images, train_labels = train.get()
    # test_images, test_labels = test.get()

    # 获取模型参数的命名空间
    arg_scope = vgg.vgg_arg_scope()

    # 创建网络
    with slim.arg_scope(arg_scope):

        '''
        2.定义占位符和网络结构
        '''
        # 输入图片
        input_images = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
        # 图片标签
        input_labels = tf.placeholder(tf.int64, shape=[None, 1])#因为后面要比较交叉熵
        # 训练还是测试？测试的时候弃权参数会设置为1.0
        #is_training = tf.placeholder(dtype=tf.bool)
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = 0.00001

        # 创建vgg16网络  如果想冻结所有层，可以指定slim.conv2d中的 trainable=False
        net, end_points = vgg.vgg_16(input_images, is_training=True, num_classes=NUM_CLASSES)
        print(end_points)#每个元素都是以vgg_16/xx命名
        # net = tf.squeeze(net, axis=[1, 2])
        # net = slim.fully_connected(net, num_outputs=NUM_CLASSES,
        #                            activation_fn=None, scope='fc8')
        # Restore only the convolutional layers: 从检查点载入当前图除了fc8层之外所有变量的参数
    params = slim.get_variables_to_restore(exclude=['vgg_16/fc8', 'global_step'])
    # 用于恢复模型  如果使用这个保存或者恢复的话，只会保存或者恢复指定的变量
    restorer = tf.train.Saver(params)
    #variables_to_train = get_tuned_variable(['fc8'])
    #load_fn = slim.assign_from_checkpoint(checkpoint_file, variables_to_train, ignore_missing_vars=True)
    # 预测标签
    pred = tf.argmax(net, axis=1)

    label_one_hot = tf.squeeze(tf.one_hot(input_labels - 1, NUM_CLASSES))
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=label_one_hot, logits=net))

    # 设置优化器
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

    # 预测结果评估
    pred = tf.reshape(pred, [-1, 1])
    correct = tf.equal(pred+1, input_labels) # 返回一个数组 表示统计预测正确或者错误
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  # 求准确率

    # 用于保存检查点文件
    save = tf.train.Saver(max_to_keep=3)

        # 恢复模型
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 检查最近的检查点文件
        ckpt = tf.train.latest_checkpoint(train_log_dir)
        if ckpt != None:
            save.restore(sess, ckpt)
            print('从上次训练保存后的模型继续训练！')
        else:
            restorer.restore(sess, checkpoint_file)
            #load_fn(sess)
            print('从官方模型加载训练！')
        '''
查看预处理之后的图片
        '''
        # imgs, labs = train.get()
        # print('原始训练图片信息：', imgs.shape, labs.shape)
        # show_img = np.array(imgs[0], dtype=np.uint8)
        # plt.imshow(show_img)
        # plt.title('Original train image')
        # plt.show()
        #
        # imgs, labs = test.get()
        # print('原始测试图片信息：', imgs.shape, labs.shape)
        # show_img = np.array(imgs[0], dtype=np.uint8)
        # plt.imshow(show_img)
        # plt.title('Original test image')
        # plt.show()
        # print(train.num)

        train.reset_num()
        test.reset_num()
        test.batch_size = 150

        print('开始训练！')
        total_cost = 0
        while train.epoch < training_epochs:
            imgs, labs = train.get()
            _, loss, global_step1, input_labels1, pred1, accuracy1 = sess.run([optimizer, cost, global_step, input_labels, pred, accuracy],
                                            feed_dict={input_images: imgs, input_labels: labs})

            total_cost += loss
            print(train.epoch, train.num)

            if train.epoch > training_epochs:#保证及时退出
                train.epoch = 1
                break
            # 打印信息
            if train.epoch % 1 == 0:
                print('Epoch {}/{}  loss {:.9f} accuracy:{:.5f}'.format(train.epoch, global_step1, loss, accuracy1))

            # 进行预测处理
            if (global_step1 % 20) == 0:
                imgs, labs = test.get()
                cost_values, accuracy_value = sess.run([cost, accuracy],
                                                       feed_dict={input_images: imgs, input_labels: labs})
                print('Epoch {}/{}  Test cost {:.9f}'.format(test.epoch, display_epoch, cost_values))
                print('准确率:', accuracy_value)
                #保存模型
                save.save(sess, os.path.join(train_log_dir, train_log_file), global_step=global_step1)
                print('Epoch:{}, global_step:{}  模型保存成功'.format(train.epoch, global_step1, training_epochs))

        print('训练完成')



if __name__ == '__main__':

    tf.reset_default_graph()

    flowers_fine_tuning()
