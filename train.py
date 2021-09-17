 # coding:utf-8
import os
import numpy as np
import tensorflow as tf
import time
start_time=time.time()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

##############################   system parameters  ###########################################

train_dir=os.getcwd()+"/trainsample/"
logs_train_dir=os.getcwd()+"/Logs/"

count = 0
for root,dirs,files in os.walk(train_dir):    
      for each in files:
             count += 1
print( "The number of samples in the training set= %d" % count)

number_conv1 = 16
number_conv2 = 32
number_conv3 = 64
number_FCL = 64

   
IMG_W=90
IMG_H=50
BATCH_SIZE=10
CAPACITY=1000+3*BATCH_SIZE
MAXIMUM_EPOCHS=500
SAMPLES=count
learning_rate=0.001
MAX_STEP=int(MAXIMUM_EPOCHS*SAMPLES/BATCH_SIZE)
print("The total training steps = %d" %MAX_STEP)



############## Tensorflow expect input which is formated in NHWC format : (BATCH,HEIGTH,WIDTH,CHANNELS) ###################

def get_files(filename):
    class_train=[]
    label_train=[]
    for train_class in os.listdir(filename):
        for pic in os.listdir(filename+train_class):
            class_train.append(filename+train_class+'/'+pic)
            label_train.append(train_class)
    temp=np.array([class_train,label_train])
    temp=temp.transpose()

    np.random.shuffle(temp)
    image_list=list(temp[:,0])
    label_list=list(temp[:,1])
    label_list=[float(i) for i in label_list]
    return image_list, label_list    

def get_batch(image,label,resize_w,resize_h,batch_size,capacity):
    #convert the list of images and labels to tensor 
    image=tf.cast(image,tf.string)
    label=tf.cast(label,tf.float32)

    input_queue=tf.train.slice_input_producer([image,label])
    label=input_queue[1]
    image_contents=tf.read_file(input_queue[0])
    image=tf.image.decode_jpeg(image_contents,channels=3)   #### channels=3 for RGB image, 1 for grayscale image
    image=tf.image.resize_images(image, [resize_w, resize_h]) 
    image=tf.image.per_image_standardization(image)

    image_batch, label_batch=tf.train.batch([image,label],batch_size = batch_size,num_threads=1,capacity = capacity)
    image_batch=tf.cast(image_batch,tf.float32)
    label_batch=tf.cast(label_batch,tf.float32)
    label_batch=tf.reshape(label_batch,[batch_size,1])
    return image_batch, label_batch

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
def regression(images,batch_size):

    with tf.variable_scope("conv1",reuse=tf.AUTO_REUSE) as scope:
        weights=tf.get_variable("weights",
				                trainable=True,
                                shape=[3,3,3,number_conv1],   ##[3,3,1,16] 1: channels=1, grayscale ; [3,3,3,16] 3: channels=3, RGB
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None))
        biases=tf.get_variable("biases",
				                trainable=True,
                                shape=[number_conv1],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer())
        conv1=tf.nn.conv2d(images,weights,strides=[1,1,1,1],padding="SAME")
        pre_activation=tf.nn.bias_add(conv1,biases)
        conv1=tf.nn.relu(pre_activation,name="conv1")


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

    #conv2
    with tf.variable_scope("conv2",reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights",
				   trainable=True,
                                   shape=[3, 3, number_conv1, number_conv2],
                                   dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None))
        biases= tf.get_variable("biases",
				trainable=True,
                                shape=[number_conv2],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer())
        conv2=tf.nn.conv2d(norm1,weights,strides=[1,1,1,1],padding="SAME")
        pre_activation=tf.nn.bias_add(conv2,biases)
        conv2=tf.nn.relu(pre_activation,name="conv2")

    # pool2 && norm2
    with tf.variable_scope("pooling2_lrn",reuse=tf.AUTO_REUSE) as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling2")
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001/9.0,beta=0.75, name='norm2')


    #conv3
    with tf.variable_scope("conv3",reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights",
				   trainable=True,
                                   shape=[3, 3, number_conv2, number_conv3],
                                   dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None))
        biases= tf.get_variable("biases",
				trainable=True,
                                shape=[number_conv3],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer())
        conv3=tf.nn.conv2d(norm2,weights,strides=[1,1,1,1],padding="SAME")
        pre_activation=tf.nn.bias_add(conv3,biases)
        conv3=tf.nn.relu(pre_activation,name="conv3")

    # pool3 && norm3
    with tf.variable_scope("pooling3_lrn",reuse=tf.AUTO_REUSE) as scope:
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling3")
        norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001/9.0,beta=0.75, name='norm3')

    
    # full-connect1
    with tf.variable_scope("fc1",reuse=tf.AUTO_REUSE) as scope:
        reshape = tf.reshape(norm3, shape=[batch_size, -1])
        dim = reshape.get_shape()[-1].value
        weights = tf.get_variable("weights",
				   trainable=True,
                                   shape=[dim,number_FCL],
                                   dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases=tf.get_variable("biases",
				trainable=True,
                                shape=[number_FCL],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer())
        pre_activation=tf.matmul(reshape,weights)+biases
        fc1=tf.nn.relu(pre_activation,name="fc1")


    # full-connect2
    with tf.variable_scope("fc2",reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights",
                                    shape=[number_FCL,1],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                shape=[1],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer())
        outputs=tf.matmul(fc1,weights)+biases
        outputs=tf.reshape(outputs,[batch_size,1])
    return outputs
##############################################      train      ########################################################

def losses(targets, outputs):
    with tf.variable_scope("loss") as scope:
        loss=tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(targets,outputs))))  #  RMSE                                                               
        #loss=tf.reduce_mean(tf.square(outputs-targets))   # MSE
        tf.summary.scalar(scope.name + "loss", loss)
    return loss

def training(loss):
    with tf.name_scope("optimizer"):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        #learning_rate=tf.train.exponential_decay(0.005,global_step,decay_steps=int(SAMPLES/BATCH_SIZE)*4,decay_rate=0.90,staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(targets, outputs):
    with tf.variable_scope("accuracy") as scope:
        correct = 1-tf.div(tf.abs(targets-outputs),targets)      
        correct = tf.cast(correct, tf.float32)
        accuracy= 100*tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + "accuracy", accuracy)
    return accuracy

def rmse(targets, outputs):      #均方根误差
    with tf.variable_scope("accuracy") as scope:
        rmse=tf.sqrt(tf.reduce_mean(tf.square(outputs-targets)))
        tf.summary.scalar(scope.name + "rmse", rmse)
    return rmse

def mse(targets,outputs):         #均方误差
    with tf.variable_scope("accuracy") as scope:
        mse=tf.reduce_mean(tf.square(outputs-targets))
        tf.summary.scalar(scope.name + "mse", mse)
    return mse

def mae(targets,outputs):          #平均绝对误差
    with tf.variable_scope("accuracy") as scope:
        mae=tf.reduce_mean(tf.abs(outputs-targets))
        tf.summary.scalar(scope.name + "mae", mae)
    return mae
########################  retrain ##################################################################

def run_retrain():

    train,train_label = get_files(train_dir)
    train_batch,train_label_batch = get_batch(train,train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    train_outputs=regression(train_batch,BATCH_SIZE)
    train_loss = losses(train_label_batch,train_outputs)
    train_op = training(train_loss)
    train_acc = evaluation(train_label_batch,train_outputs)
    train_rmse = rmse(train_label_batch, train_outputs)
    train_mse=mse(train_label_batch, train_outputs)
    train_mae=mae(train_label_batch, train_outputs)


    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)
    saver = tf.train.Saver()

    listout_t = []
    listtar_t = []
    list_t = []
    rmse_t = []
    MAE_t=[]
    MSE_t=[]
    try:

        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break                
            _,tra_loss,tra_acc,tra_out,tra_label_batch,tra_rmse,tra_mse,tra_mae = sess.run([train_op,train_loss,train_acc,train_outputs,train_label_batch,train_rmse,train_mse,train_mae])
                
            if step %  50 == 0:
                print('Step %d,train loss = %.4f,train accuracy = %.4f%%'%(step,tra_loss,tra_acc))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str,step)
        
            if step % 100 ==0 or (step +1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step = step)

            # list_num_2000 = []
            # for i in range(2000 - 20, 2000):
            #     list_num_2000.append(i)
            #
            # if (step + 1) in list_num_2000:
            #     # if (step+1) == MAX_STEP:
            #     # 将train_outputs中的内容添加到列表中
            #     listout_t.append(tra_out)
            #     listtar_t.append(tra_label_batch)
            #     rmse_t.append(tra_rmse)
            #     MAE_t.append(tra_mae)
            #     MSE_t.append(tra_mse)
            #
            #     listtraout = np.array(listout_t)
            #     listtratar = np.array(listtar_t)
            #     listtrarmse = np.array(np.mean(rmse_t))
            #     listmse = np.array(np.mean(MSE_t))
            #     listmae = np.array(np.mean(MAE_t))
            #
            #     listtrain = np.hstack((listtraout.reshape([-1, 1]), listtratar.reshape([-1, 1])))
            #
            #     np.savetxt('train2000.dat', listtrain, fmt="%.4f")  # 预测值  真实值
            #     np.savetxt('rmse_train2000.txt', listtrarmse.reshape([-1, 1]), fmt="%.4f")
            #     np.savetxt('MSE_train2000.txt', listmse.reshape([-1, 1]), fmt="%.4f")
            #     np.savetxt('MAE_train2000.txt', listmae.reshape([-1, 1]), fmt="%.4f")
            #
            # list_num_4000 = []
            # for i in range(4000 - 20, 4000):
            #     list_num_4000.append(i)
            #
            # if (step + 1) in list_num_4000:
            #     # if (step+1) == MAX_STEP:
            #     # 将train_outputs中的内容添加到列表中
            #     listout_t.append(tra_out)
            #     listtar_t.append(tra_label_batch)
            #     rmse_t.append(tra_rmse)
            #     MAE_t.append(tra_mae)
            #     MSE_t.append(tra_mse)
            #
            #     listtraout = np.array(listout_t)
            #     listtratar = np.array(listtar_t)
            #     listtrarmse = np.array(np.mean(rmse_t))
            #     listmse = np.array(np.mean(MSE_t))
            #     listmae = np.array(np.mean(MAE_t))
            #
            #     listtrain = np.hstack((listtraout.reshape([-1, 1]), listtratar.reshape([-1, 1])))
            #
            #     np.savetxt('train4000.dat', listtrain, fmt="%.4f")  # 预测值  真实值
            #     np.savetxt('rmse_train4000.txt', listtrarmse.reshape([-1, 1]), fmt="%.4f")
            #     np.savetxt('MSE_train4000.txt', listmse.reshape([-1, 1]), fmt="%.4f")
            #     np.savetxt('MAE_train4000.txt', listmae.reshape([-1, 1]), fmt="%.4f")
            #
            # list_num_6000 = []
            # for i in range(6000 - 20, 6000):
            #     list_num_6000.append(i)
            #
            # if (step + 1) in list_num_6000:
            #     # if (step+1) == MAX_STEP:
            #     # 将train_outputs中的内容添加到列表中
            #     listout_t.append(tra_out)
            #     listtar_t.append(tra_label_batch)
            #     rmse_t.append(tra_rmse)
            #     MAE_t.append(tra_mae)
            #     MSE_t.append(tra_mse)
            #
            #     listtraout = np.array(listout_t)
            #     listtratar = np.array(listtar_t)
            #     listtrarmse = np.array(np.mean(rmse_t))
            #     listmse = np.array(np.mean(MSE_t))
            #     listmae = np.array(np.mean(MAE_t))
            #
            #     listtrain = np.hstack((listtraout.reshape([-1, 1]), listtratar.reshape([-1, 1])))
            #
            #     np.savetxt('train6000.dat', listtrain, fmt="%.4f")  # 预测值  真实值
            #     np.savetxt('rmse_train6000.txt', listtrarmse.reshape([-1, 1]), fmt="%.4f")
            #     np.savetxt('MSE_train6000.txt', listmse.reshape([-1, 1]), fmt="%.4f")
            #     np.savetxt('MAE_train6000.txt', listmae.reshape([-1, 1]), fmt="%.4f")
            #
            # list_num_8000 = []
            # for i in range(8000 - 20, 8000):
            #     list_num_8000.append(i)
            #
            # if (step + 1) in list_num_8000:
            #     # if (step+1) == MAX_STEP:
            #     # 将train_outputs中的内容添加到列表中
            #     listout_t.append(tra_out)
            #     listtar_t.append(tra_label_batch)
            #     rmse_t.append(tra_rmse)
            #     MAE_t.append(tra_mae)
            #     MSE_t.append(tra_mse)
            #
            #     listtraout = np.array(listout_t)
            #     listtratar = np.array(listtar_t)
            #     listtrarmse = np.array(np.mean(rmse_t))
            #     listmse = np.array(np.mean(MSE_t))
            #     listmae = np.array(np.mean(MAE_t))
            #
            #     listtrain = np.hstack((listtraout.reshape([-1, 1]), listtratar.reshape([-1, 1])))
            #
            #     np.savetxt('train8000.dat', listtrain, fmt="%.4f")  # 预测值  真实值
            #     np.savetxt('rmse_train8000.txt', listtrarmse.reshape([-1, 1]), fmt="%.4f")
            #     np.savetxt('MSE_train8000.txt', listmse.reshape([-1, 1]), fmt="%.4f")
            #     np.savetxt('MAE_train8000.txt', listmae.reshape([-1, 1]), fmt="%.4f")
            #
            # list_num_10000 = []
            # for i in range(10000 - 20, 10000):
            #     list_num_10000.append(i)
            #
            # if (step + 1) in list_num_10000:
            #     # if (step+1) == MAX_STEP:
            #     # 将train_outputs中的内容添加到列表中
            #     listout_t.append(tra_out)
            #     listtar_t.append(tra_label_batch)
            #     rmse_t.append(tra_rmse)
            #     MAE_t.append(tra_mae)
            #     MSE_t.append(tra_mse)
            #
            #     listtraout = np.array(listout_t)
            #     listtratar = np.array(listtar_t)
            #     listtrarmse = np.array(np.mean(rmse_t))
            #     listmse = np.array(np.mean(MSE_t))
            #     listmae = np.array(np.mean(MAE_t))
            #
            #     listtrain = np.hstack((listtraout.reshape([-1, 1]), listtratar.reshape([-1, 1])))
            #
            #     np.savetxt('train10000.dat', listtrain, fmt="%.4f")  # 预测值  真实值
            #     np.savetxt('rmse_train10000.txt', listtrarmse.reshape([-1, 1]), fmt="%.4f")
            #     np.savetxt('MSE_train10000.txt', listmse.reshape([-1, 1]), fmt="%.4f")
            #     np.savetxt('MAE_train10000.txt', listmae.reshape([-1, 1]), fmt="%.4f")

            # list_num_12000 = []
            # for i in range(12000 - 20, 12000):
            #     list_num_12000.append(i)
            #
            # if (step + 1) in list_num_12000:
            #     # if (step+1) == MAX_STEP:
            #     # 将train_outputs中的内容添加到列表中
            #     listout_t.append(tra_out)
            #     listtar_t.append(tra_label_batch)
            #     rmse_t.append(tra_rmse)
            #     MAE_t.append(tra_mae)
            #     MSE_t.append(tra_mse)
            #
            #     listtraout = np.array(listout_t)
            #     listtratar = np.array(listtar_t)
            #     listtrarmse = np.array(np.mean(rmse_t))
            #     listmse = np.array(np.mean(MSE_t))
            #     listmae = np.array(np.mean(MAE_t))
            #
            #     listtrain = np.hstack((listtraout.reshape([-1, 1]), listtratar.reshape([-1, 1])))
            #
            #     np.savetxt('train12000.dat', listtrain, fmt="%.4f")  # 预测值  真实值
            #     np.savetxt('rmse_train12000.txt', listtrarmse.reshape([-1, 1]), fmt="%.4f")
            #     np.savetxt('MSE_train12000.txt', listmse.reshape([-1, 1]), fmt="%.4f")
            #     np.savetxt('MAE_train12000.txt', listmae.reshape([-1, 1]), fmt="%.4f")

            list_num=[]
            for i in range(MAX_STEP-20,MAX_STEP):
                list_num.append(i)

            if (step+1) in list_num:
            #if (step+1) == MAX_STEP:
                #将train_outputs中的内容添加到列表中
                listout_t.append(tra_out)
                listtar_t.append(tra_label_batch)
                rmse_t.append(tra_rmse)
                MAE_t.append(tra_mae)
                MSE_t.append(tra_mse)
                
                listtraout=np.array(listout_t)
                listtratar=np.array(listtar_t)
                listtrarmse=np.array(np.mean(rmse_t))
                listmse=np.array(np.mean(MSE_t))
                listmae=np.array(np.mean(MAE_t))

                listtrain=np.hstack((listtraout.reshape([-1,1]),listtratar.reshape([-1,1])))

                np.savetxt('train.dat',listtrain,fmt="%.4f")       #预测值  真实值
                np.savetxt('rmse_train.txt',listtrarmse.reshape([-1,1]),fmt="%.4f")
                np.savetxt('MSE_train.txt',listmse.reshape([-1,1]),fmt="%.4f")
                np.savetxt('MAE_train.txt',listmae.reshape([-1,1]),fmt="%.4f")


    except tf.errors.OutOfRangeError:
        print('Done training epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

run_retrain()
end_time=time.time()
print("time:%d"%((end_time-start_time)/60)+"min")

          




