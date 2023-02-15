# coding:utf-8
import os
import numpy as np
import tensorflow as tf
import time
import random
import shutil

start_time = time.time()
# np.random.seed(1)
# tf.set_random_seed(1)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Randomly allocate training set and test set
def random_select():
    list_png = [float(i) for i in os.listdir(os.getcwd() + '\\512png')]
    print(list_png)
    list_train_png = random.sample(list_png, 200)
    print(list_train_png)  # 200 random images data in the train set
    # print(len(list_train_png))
    list_pre_png = []
    for png in list_png:
        if png not in list_train_png:
            list_pre_png.append(png)
    print(list_pre_png)  # 312 random images data in the test set

    local_workpath = os.getcwd()
    list_train_add = []
    list_pre_add = []
    for trainexample in list_train_png:
        list_train_add.append(local_workpath + '\\512png\\' + str(trainexample))
    for preexample in list_pre_png:
        list_pre_add.append(local_workpath + '\\512png\\' + str(preexample))
    print(list_train_add)
    print(list_pre_add)
    if not os.path.exists('trainsample'):
        os.mkdir('trainsample')
    for tra_add in list_train_add:
        shutil.move(tra_add, os.getcwd() + '\\' + 'trainsample')
    if not os.path.exists('predictsample'):
        os.mkdir('predictsample')
    for pre_add, pre_pngs in zip(list_pre_add, list_pre_png):
        shutil.move(pre_add + '\\' + str(pre_pngs) + '.png', os.getcwd() + '\\' + 'predictsample')
#random_select()
##############################   system parameters  ###########################################
train_dir = os.getcwd() + "/trainsample/"
logs_train_dir = os.getcwd() + "/Logs/"
count = 0
for root, dirs, files in os.walk(train_dir):
    for each in files:
        count += 1
print("The number of samples in the training set= %d" % count)

# number_conv1 = 16
# number_conv2 = 32
# number_conv3 =80
# number_FCL = 100

number_conv1 = 16
number_conv2 = 32
number_conv3 =64
number_FCL = 80
# IMG_W = 90
# IMG_H = 50
IMG_W = 90
IMG_H = 50
BATCH_SIZE =10

CAPACITY = 1000 + 3 * BATCH_SIZE
MAXIMUM_EPOCHS = 505
SAMPLES = count
learning_rate = 0.001
#learning_rate=tf.train.exponential_decay()
MAX_STEP = int(MAXIMUM_EPOCHS * SAMPLES / BATCH_SIZE)
print("The total training steps = %d" % MAX_STEP)

############## Tensorflow expect input which is formated in NHWC format : (BATCH,HEIGTH,WIDTH,CHANNELS) ###################

def get_files(filename):
    class_train = []
    label_train = []
    for train_class in os.listdir(filename):
        for pic in os.listdir(filename + train_class):
            class_train.append(filename + train_class + '/' + pic)
            label_train.append(train_class)
    temp = np.array([class_train, label_train])
    temp = temp.transpose()

    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [float(i) for i in label_list]
    print(image_list)
    print(label_list)
    return image_list, label_list

def get_batch(image, label, resize_w, resize_h, batch_size, capacity):
    # convert the list of images and labels to tensor
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.float32)

    input_queue = tf.train.slice_input_producer([image, label],shuffle=False)
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)  #### channels=3 for RGB image, 1 for grayscale image
    image = tf.image.resize_images(image, [resize_w, resize_h])
    #image=tf.image.resize_image_with_crop_or_pad(image,resize_w,resize_h)
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=1, capacity=capacity)
    #image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=1, capacity=capacity)
    image_batch = tf.cast(image_batch, tf.float32)
    label_batch = tf.cast(label_batch, tf.float32)
    label_batch = tf.reshape(label_batch, [batch_size, 1])
    return image_batch, label_batch

def seed_tensorflow(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
seed_tensorflow(1)

########################################     CNN model used for regression  ##################################################
# Description:                                                                                                               #
# 1. CNN :        c16c32c64f64                                                                                               #
# 2. Parameters:  batch_size=300  learning_rate=0.0001   number of maximum epochs=300                                        #
# 3. Setup: kernel size 3*3 with a stride of 1. Each convolutionnal layer was followed by a ReLU function                    #
#           and a max-pooling layer of size 2*2 with a stride of 2.  The same padding was used after the first               #
#           convolutional layer to preserve the image size. We included one fully-connected layer of size ranging            #
#           from 4 to 2024. As we performed regression, we did not include the ReLU function at the end of the final layer.  #
#           The Adam optimizer was used to minimized the mean squared error.                                                 #
##############################################################################################################################
def regression(images, batch_size):
    with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE) as scope:
        # weights为卷积核，要求是一个张量
        weights = tf.get_variable("weights",
                                  trainable=True,
                                  shape=[3, 3, 3, number_conv1],   #shape为 [ filter_height, filter_width, in_channel, out_channels ]
                                  #[3,3,1,16] 1: channels=1, grayscale ; [3,3,3,16] 3: channels=3, RGB
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=1)
                                  #initializer=tf.truncated_normal_initializer(stddev=0.001, dtype=tf.float32,seed=1)     #截断正态分布初始化器
                                  )
        biases = tf.get_variable("biases",
                                 trainable=True,
                                 shape=[number_conv1],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer()     #正态分布
                                 #initializer = tf.constant_initializer(0.1)      #常量初始化器
                                 )
        conv1 = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv1, biases)
        conv1 = tf.nn.relu(pre_activation, name="conv1")

        weights1 = tf.get_variable("weights1",
                                   trainable=True,
                                   shape=[3, 3, number_conv1, number_conv1],
                                   ##[3,3,1,16] 1: channels=1, grayscale ; [3,3,3,16] 3: channels=3, RGB
                                   dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=1))
        biases1 = tf.get_variable("biases1",
                                  trainable=True,
                                  shape=[number_conv1],
                                  dtype=tf.float32,
                                  initializer=tf.random_normal_initializer() )
        conv1_2 = tf.nn.conv2d(conv1, weights1, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation1_2 = tf.nn.bias_add(conv1_2, biases1)
        conv1_2 = tf.nn.relu(pre_activation1_2, name="conv1_2")

    # pool1 && norm1
    with tf.variable_scope("pooling1_lrn", reuse=tf.AUTO_REUSE) as scope:
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pooling1")
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope("conv2", reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights",
                                  trainable=True,
                                  shape=[3, 3, number_conv1, number_conv2],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=1))
        biases = tf.get_variable("biases",
                                 trainable=True,
                                 shape=[number_conv2],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer() )
        conv2 = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv2, biases)
        conv2 = tf.nn.relu(pre_activation, name="conv2")

    # pool2 && norm2
    with tf.variable_scope("pooling2_lrn", reuse=tf.AUTO_REUSE) as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding="SAME", name="pooling2")
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # conv3
    with tf.variable_scope("conv3", reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights",
                                  trainable=True,
                                  shape=[3, 3, number_conv2, number_conv3],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=1))
        biases = tf.get_variable("biases",
                                 trainable=True,
                                 shape=[number_conv3],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer() )
        conv3 = tf.nn.conv2d(norm2, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv3, biases)
        conv3 = tf.nn.relu(pre_activation, name="conv3")

    # pool3 && norm3
    with tf.variable_scope("pooling3_lrn", reuse=tf.AUTO_REUSE) as scope:
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding="SAME", name="pooling3")
        norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
        #droped_3 = tf.nn.dropout(pool3, keep_prob=0.8,seed=1)

    # full-connect1
    with tf.variable_scope("fc1", reuse=tf.AUTO_REUSE) as scope:
        reshape = tf.reshape(norm3, shape=[batch_size, -1])
        dim = reshape.get_shape()[-1].value
        weights = tf.get_variable("weights",
                                  trainable=True,
                                  shape=[dim, number_FCL],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 trainable=True,
                                 shape=[number_FCL],
                                 dtype=tf.float32,
                                 initializer = tf.constant_initializer(0.1))
        pre_activation = tf.matmul(reshape, weights) + biases
        fc1 = tf.nn.relu(pre_activation, name="fc1")

    # full-connect2
    with tf.variable_scope("fc2", reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights",
                                  shape=[number_FCL, 1],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[1],
                                 dtype=tf.float32,
                                 initializer = tf.constant_initializer(0.1))
        outputs = tf.matmul(fc1, weights) + biases
        outputs = tf.reshape(outputs, [batch_size, 1])

    return outputs

##############################################      train      ########################################################

def losses(targets, outputs):
    with tf.variable_scope("loss") as scope:
        loss = tf.sqrt(tf.reduce_mean(tf.square(
            tf.subtract(targets, outputs))))  # RMSE
        # loss=tf.reduce_mean(tf.square(outputs-targets))   # MSE
        tf.summary.scalar(scope.name + "loss", loss)
    return loss

def training(loss):
    with tf.name_scope("optimizer"):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # learning_rate=tf.train.exponential_decay(0.005,global_step,decay_steps=int(SAMPLES/BATCH_SIZE)*4,decay_rate=0.90,staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #optimizer=tf.train.RMSPropOptimizer(0.9).minimize(loss)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(targets, outputs):
    with tf.variable_scope("accuracy") as scope:
        correct = 1 - tf.div(tf.abs(targets - outputs), targets)
        correct = tf.cast(correct, tf.float32)
        accuracy = 100 * tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + "accuracy", accuracy)
    return accuracy

def rmse(targets, outputs):  # 均方根误差
    with tf.variable_scope("accuracy") as scope:
        rmse = tf.sqrt(tf.reduce_mean(tf.square(outputs - targets)))
        tf.summary.scalar(scope.name + "rmse", rmse)
    return rmse

def mse(targets, outputs):  # 均方误差
    with tf.variable_scope("accuracy") as scope:
        mse = tf.reduce_mean(tf.square(outputs - targets))
        tf.summary.scalar(scope.name + "mse", mse)
    return mse

def mae(targets, outputs):  # 平均绝对误差
    with tf.variable_scope("accuracy") as scope:
        mae = tf.reduce_mean(tf.abs(outputs - targets))
        tf.summary.scalar(scope.name + "mae", mae)
    return mae
def deviation_index(pre,true,number):
    deviation=abs(pre-true)   #对应元素差值
    print(deviation)             #差值列表
    list_temp=[]
    #print(type(deviation)) #<class 'numpy.ndarray'>
    for i in deviation:
        if(i>number):
            list_temp.append(i)
    print(list_temp)
    b=np.arange(len(deviation))  #生成与deviation长度相等的数组
    index=b[deviation>number]
    print(index)
    list_new=[]
    for j in index:
        list_new.append(true[j])
    #print(list_new)
    return list_new

########################  retrain ##################################################################

def run_retrain():
    train, train_label = get_files(train_dir)
    train_batch, train_label_batch = get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    train_outputs = regression(train_batch, BATCH_SIZE)
    train_loss = losses(train_label_batch, train_outputs)  #训练损失
    train_op = training(train_loss)
    train_acc = evaluation(train_label_batch, train_outputs)
    train_rmse = rmse(train_label_batch, train_outputs)
    train_mse = mse(train_label_batch, train_outputs)
    train_mae = mae(train_label_batch, train_outputs)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    saver = tf.train.Saver()

    listout_t = []
    listtar_t = []
    list_loss= []
    list_epoch=[]
    list_acc=[]
    rmse_t = []
    MAE_t = []
    MSE_t = []
    try:

        for step in np.arange(MAX_STEP):    #np.arange()返回一个固定长度的列表
            if coord.should_stop():
                break
            _, tra_loss, tra_acc, tra_out, tra_label_batch, tra_rmse, tra_mse, tra_mae = sess.run(
                [train_op, train_loss, train_acc, train_outputs, train_label_batch, train_rmse, train_mse, train_mae])
            #print("****")
            if step % 50 == 0:
                print('Step %d,train loss = %.4f,train accuracy = %.4f%%' % (step, tra_loss, tra_acc))
                list_epoch.append(0.05*step)
                list_loss.append(tra_loss)
                list_acc.append(tra_acc)
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 100 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            list_num = []
            for i in range(MAX_STEP -int(count/BATCH_SIZE), MAX_STEP):
                list_num.append(i)

            if (step + 1) in list_num:
                # if (step+1) == MAX_STEP:
                # 将train_outputs中的内容添加到列表中
                listout_t.append(tra_out)
                listtar_t.append(tra_label_batch)
                rmse_t.append(tra_rmse)
                MAE_t.append(tra_mae)
                MSE_t.append(tra_mse)

                listtraout = np.array(listout_t)  #预测值
                listtratar = np.array(listtar_t)  #实际值
                # list_true_active=deviation_index(listtraout,listtratar,2)
                # print(list_true_active)
                # listtrarmse = np.array(np.mean(rmse_t))
                # listmse = np.array(np.mean(MSE_t))
                # listmae = np.array(np.mean(MAE_t))

                listtrain = np.hstack((listtraout.reshape([-1, 1]), listtratar.reshape([-1, 1])))

                np.savetxt('train.dat', listtrain, fmt="%.4f")  # 预测值  真实值
                # np.savetxt('rmse_train.txt', listtrarmse.reshape([-1, 1]), fmt="%.4f")
                # np.savetxt('MSE_train.txt', listmse.reshape([-1, 1]), fmt="%.4f")
                # np.savetxt('MAE_train.txt', listmae.reshape([-1, 1]), fmt="%.4f")


    except tf.errors.OutOfRangeError:
        print('Done training epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
    print(list_loss)       #y轴
    print(len(list_loss))
    print(list_epoch)      #x轴
    print(len(list_epoch))
    with open("loss.txt",'w',encoding="utf-8") as f:
        for index,item,accuracy in zip(list_epoch,list_loss,list_acc):
            f.write(str(index))
            f.write("  ")
            f.write(str(item))
            f.write("  ")
            f.write(str(accuracy))
            f.write("\n")
    f.close()
run_retrain()
end_time = time.time()
print("time:%d" % ((end_time - start_time) / 60) + "min")






