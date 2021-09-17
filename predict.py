# coding:utf-8
import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#####################################################################################################

pwd = os.getcwd()
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")

print(father_path)
predict_dir = "images"
logs_dir = "./Logs"#father_path + "/Logs/"#"/cnb_wenjian_huatu_redaolv/Logs/"
print(logs_dir)

count = 0
for root, dirs, files in os.walk(predict_dir):
    for each in files:
        count += 1
print(count)

number_conv1 = 16
number_conv2 = 32
number_conv3 = 64
number_FCL = 64

#number_conv1 = 32
#number_conv2 = 64
#number_conv3 = 128
#number_FCL = 128

IMG_W = 90
IMG_H = 50
BATCH_SIZE = 1
CAPACITY = 1000 + 3 * BATCH_SIZE
MAXIMUM_EPOCHS = 1
SAMPLES = count
MAX_STEP = int(MAXIMUM_EPOCHS * SAMPLES / BATCH_SIZE)


def get_files(filename):
    pic_predict = []
    name = []
    for pic in os.listdir(filename):
        pic_predict.append(filename + '/' + pic)
        name.append('.'.join([pic.split('.', -1)[0],pic.split('.', -1)[1]]))
    print(name)
    temp = np.array([pic_predict, name])
    temp = temp.transpose()
    image_list = list(temp[:, 0])
    name_list = list(temp[:, 1])
    name_list = [float(i) for i in name_list]
    return image_list, name_list


def get_batch(image, name, resize_w, resize_h, batch_size, capacity):
    # convert the list of images and labels to tensor
    image = tf.cast(image, tf.string)
    name = tf.cast(name, tf.float32)
    input_queue = tf.train.slice_input_producer(([image, name]), shuffle=False, num_epochs=MAXIMUM_EPOCHS)
    name = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)  #### channels=3 for RGB image, 1 for grayscale image
    image = tf.image.resize_images(image, [resize_w, resize_h])  #### used for grayscale image
    image = tf.image.per_image_standardization(image)
    image_batch, name_batch = tf.train.batch([image, name], batch_size=batch_size, num_threads=10, capacity=capacity)
    image_batch = tf.cast(image_batch, tf.float32)
    name_batch = tf.cast(name_batch, tf.float32)
    name_batch = tf.reshape(name_batch, [batch_size, 1])
    return image_batch, name_batch


########################################     Design CNN  for regression     ##########################################

def regression(images, batch_size):
    # conv1, shape=[kernel_]
    with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights",
                                  trainable=False,
                                  shape=[3, 3, 3, number_conv1],
                                  ##[3,3,1,16] 1: channels=1, grayscale ; [3,3,3,16] 3: channels=3, RGB
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None))
        biases = tf.get_variable("biases",
                                 trainable=False,
                                 shape=[number_conv1],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer())
        conv1 = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv1, biases)
        conv1 = tf.nn.relu(pre_activation, name="conv1")
        weights1=tf.get_variable("weights1",
                                trainable=True,
                                shape=[3,3,number_conv1,number_conv1],   ##[3,3,1,16] 1: channels=1, grayscale ; [3,3,3,16] 3: channels=3, RGB
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None))
        biases1=tf.get_variable("biases1",
                                trainable=True,
                                shape=[number_conv1],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer())
        conv1_2=tf.nn.conv2d(conv1,weights1,strides=[1,1,1,1],padding="SAME")
        pre_activation1_2=tf.nn.bias_add(conv1_2,biases1)
        conv1_2=tf.nn.relu(pre_activation1_2,name="conv1_2")

    #pool1 && norm1
    with tf.variable_scope("pooling1_lrn",reuse=tf.AUTO_REUSE) as scope:
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding="SAME", name="pooling1")
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,beta=0.75, name='norm1')
    # conv2
    with tf.variable_scope("conv2", reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights",
                                  trainable=False,
                                  shape=[3, 3, number_conv1, number_conv2],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None))
        biases = tf.get_variable("biases",
                                 trainable=False,
                                 shape=[number_conv2],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer())
        conv2 = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv2, biases)
        conv2 = tf.nn.relu(pre_activation, name="conv2")
    # pool2 && norm2
    with tf.variable_scope("pooling2_lrn", reuse=tf.AUTO_REUSE) as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling2")
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001/9.0,beta=0.75, name='norm2')

    # conv3
    with tf.variable_scope("conv3", reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights",
                                  trainable=False,
                                  shape=[3, 3, number_conv2, number_conv3],
                                  dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None))
        biases = tf.get_variable("biases",
                                 trainable=False,
                                 shape=[number_conv3],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer())
        conv3 = tf.nn.conv2d(norm2, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv3, biases)
        conv3 = tf.nn.relu(pre_activation, name="conv3")
    # pool3 && norm3
    with tf.variable_scope("pooling3_lrn", reuse=tf.AUTO_REUSE) as scope:
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling3")
        norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001/9.0,beta=0.75, name='norm3')

    # full-connect1
    with tf.variable_scope("fc1", reuse=tf.AUTO_REUSE) as scope:
        reshape = tf.reshape(norm3, shape=[batch_size, -1])
        dim = reshape.get_shape()[-1].value
        weights = tf.get_variable("weights",
                                  trainable=False,
                                  shape=[dim, number_FCL],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 trainable=False,
                                 shape=[number_FCL],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer())
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
                                 initializer=tf.random_normal_initializer())
        outputs = tf.matmul(fc1, weights) + biases
        outputs = tf.reshape(outputs, [batch_size, 1])
    return outputs

def rmse(targets, outputs):
    with tf.variable_scope("accuracy") as scope:
        rmse=tf.sqrt(tf.reduce_mean(tf.square(outputs-targets)))
    return rmse

def mse(targets, outputs):
    with tf.variable_scope("accuracy") as scope:
        mse=tf.reduce_mean(tf.square(outputs-targets))
    return mse

def mae(targets, outputs):
    with tf.variable_scope("accuracy") as scope:
        mae=tf.reduce_mean(tf.abs(outputs-targets))
    return mae

def run_predict():
    img_input, name_input = get_files(predict_dir)
    img_batch, name_batch = get_batch(img_input, name_input, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    print(name_batch)
    predict = regression(img_batch, BATCH_SIZE)
    rmse_predict = rmse(name_batch,predict)
    print(predict)
    mse_predict=mse(name_batch,predict)
    mae_predict=mae(name_batch,predict)

    L1 = []
    L2 = []
    L3 = []
    L4=[]
    L5=[]

    sess = tf.Session()
    coord = tf.train.Coordinator()
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.train.Saver()
    print("---------------------------")
    print("model path:", logs_dir)
    ##saver.restore
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("Restore model...")
        print(ckpt.model_checkpoint_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
        print("successful to restore:",  ckpt.model_checkpoint_path)
    print("---------------------------")

    ##ckpt = tf.train.get_checkpoint_state(logs_dir)
    ##global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    ###saver.restore(sess, ckpt.model_checkpoint_path)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            nlist, predlist,pred_rmse,pre_mse,pre_mae = sess.run([name_batch, predict,rmse_predict,mse_predict,mae_predict])
            L1.append(nlist)
            L2.append(predlist)
            L3.append(pred_rmse)
            L4.append(pre_mse)
            L5.append(pre_mae)
            print('Step %d, the number of predicted images = %d' % (step, BATCH_SIZE * step))

        list1 = np.array(L1)
        #print(L1)            [array([[100.]], dtype=float32), array([[100.]], dtype=float32)]
        #print(nlist)         [[99.]]
        list2 = np.array(L2)
        rmse_mean = np.mean(L3)
        mse_mean=np.mean(L4)
        mae_mean=np.mean(L5)
        listtrarmse = np.array(rmse_mean)
        listmse=np.array(mse_mean)
        listmae=np.array(mae_mean)

        listprediction = np.hstack((list1.reshape([-1, 1]), list2.reshape([-1, 1])))          #拼接数组的方法 np.hstack():在水平方向上平铺
        #print(listprediction)
        np.savetxt('prediction.dat', listprediction, fmt="%.4f %.4f")
        #np.savetxt('rmse_pred.txt', listtrarmse.reshape([-1,1]), fmt="%.4f")
        # np.savetxt('mse_pred', listmse.reshape([-1, 1]), fmt="%.4f")
        # np.savetxt('mae_pred', listmae.reshape([-1, 1]), fmt="%.4f")
    except tf.errors.OutOfRangeError:
        print('Done training epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


run_predict()
