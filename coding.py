import tensorflow as tf
def inference(image,batch_size,n_classes):
    with tf.variable_scope("conv1_lrn") as scope:
        tf.summary.image("image",image,max_outputs=10)
        weights = tf.get_variable("weights",shape=[11,11,3,96],dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01,
                                                                              dtype=tf.float32))
        biases = tf.get_variable("biases",shape=[96],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(image,weights,strides=[1,4,4,1],padding="SAME")
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(pre_activation,name=scope.name)
        norm1 = tf.nn.lrn(conv1,depth_radius=5,bias=2.0,alpha=0.0001,beta=0.75,name="norm1")

    with tf.variable_scope("pooling1") as scope:
        pool1 = tf.nn.max_pool(norm1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID",name="pooling1")

    with tf.variable_scope("conv2") as scope:
        weights = tf.get_variable("weights",shape=[5,5,96,256],dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01,
                                                                               dtype=tf.float32))
        biases = tf.get_variable("biases",shape=[256],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool1,weights,strides=[1,1,1,1],padding="SAME")
        pre_activation = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(pre_activation,name="conv2")

    with tf.variable_scope("pooling2_lrn") as scope:
        norm2 = tf.nn.lrn(conv2,depth_radius=5,bias=2.0,alpha=0.0001,beta=0.75,name="norm2")
        pool2 = tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID",name="pooling2")

    with tf.variable_scope("conv3") as scope:
        weights = tf.get_variable("weights",shape=[3,3,256,384],dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01,
                                                                              dtype=tf.float32))
        biases = tf.get_variable("biases",shape=[384],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool2,weights,strides=[1,1,1,1],padding="SAME")
        pre_activation = tf.nn.bias_add(conv,biases)
        conv3 = tf.nn.relu(pre_activation,name="conv3")

    with tf.variable_scope("conv4") as scope:
        weights = tf.get_variable("weights",shape=[3,3,384,384],dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01,
                                                                              dtype=tf.float32))
        biases = tf.get_variable("biases", shape=[384], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv,biases)
        conv4 = tf.nn.relu(pre_activation, name="conv4")

    with tf.variable_scope("conv5") as scope:
        weights = tf.get_variable("weights",shape=[3,3,384,256],dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01,
                                                                              dtype=tf.float32))
        biases = tf.get_variable("biases", shape=[256], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv4, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv,biases)
        conv5 = tf.nn.relu(pre_activation, name="conv5")

    with tf.variable_scope("pooling6") as scope:
        pool6 = tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID",name="pooling6")

    with tf.variable_scope("local7") as scope:
        reshape = tf.reshape(pool6,shape=[batch_size,-1])
        dim = reshape.get_shape()[1].value  #------------------------------------------------------------------------
        weights = tf.get_variable("weights", shape=[dim,4096], dtype=tf.float32,
                                            initializer=tf.truncated_normal_initializer(stddev=0.005,
                                                                                        dtype=tf.float32))
        biases = tf.get_variable("biases", shape=[4096], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        local7 = tf.nn.relu(tf.matmul(reshape,weights)+biases,name="loca7")
        local7 = tf.nn.dropout(local7,keep_prob=0.5)

    with tf.variable_scope("local8") as scope:
        weights = tf.get_variable("weights", shape=[4096, 4096], dtype=tf.float32,
                                            initializer=tf.truncated_normal_initializer(stddev=0.005,
                                                                                        dtype=tf.float32))
        biases = tf.get_variable("biases", shape=[4096], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        local8 = tf.nn.relu(tf.matmul(local7, weights) + biases, name="loca8")
        local8 = tf.nn.dropout(local8, keep_prob=0.5)

    with tf.variable_scope("softmax_linear") as scope:
        weights = tf.get_variable("weights", shape=[4096, n_classes], dtype=tf.float32,
                                            initializer=tf.truncated_normal_initializer(stddev=0.005,
                                                                                        dtype=tf.float32))
        biases = tf.get_variable("biases", shape=[n_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local8, weights),biases, name="softmax_linear")#----..,,,,!!!
        tf.summary.histogram("w",softmax_linear)

        return softmax_linear

def losses(logits,labels):
    with  tf.variable_scope("loss")  as scope:
        #labels = tf.cast(labels,tf.int32)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,
                                                                       name="xebtropy_per_example")
        loss = tf.reduce_mean(cross_entropy,name="loss")

        tf.summary.scalar(scope.name+"loss",loss)
    return loss


def trainning(loss,learning_rate):
    with tf.name_scope("optimizer"):
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0,name="global_step",trainable=False)
        train_op = optimizer.minimize(loss,global_step=global_step)

    return train_op

def evaluation(logists,lables):
    with tf.variable_scope("accurancy") as  scope:

        correct = tf.nn.in_top_k(logists,lables,1)
        correct = tf.cast(correct,tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+"/accuracy",accuracy)
    return accuracy








import tensorflow as tf
def inference(image,batch_size,n_classes):
    with tf.variable_scope("conv1_lrn") as scope:
        weights = tf.get_variable("weights",shape=[11,11,3,96],dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01,
                                                                              dtype=tf.float32))
        biases = tf.get_variable("biases",shape=[96],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(image,weights,strides=[1,4,4,1],padding="SAME")
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(pre_activation,name=scope.name)
        norm1 = tf.nn.lrn(conv1,depth_radius=5,bias=2.0,alpha=0.0001,beta=0.75,name="norm1")

    with tf.variable_scope("local7") as scope:
        reshape = tf.reshape(norm1,shape=[batch_size,-1])
        dim = reshape.get_shape()[1].value  #------------------------------------------------------------------------
        weights = tf.get_variable("weights", shape=[dim,500], dtype=tf.float32,
                                            initializer=tf.truncated_normal_initializer(stddev=0.005,
                                                                                        dtype=tf.float32))
        biases = tf.get_variable("biases", shape=[500], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        local7 = tf.nn.relu(tf.matmul(reshape,weights)+biases,name="loca7")
        local7 = tf.nn.dropout(local7,keep_prob=0.5)

    with tf.variable_scope("softmax_linear") as scope:
        weights = tf.get_variable("weights", shape=[500, n_classes], dtype=tf.float32,
                                            initializer=tf.truncated_normal_initializer(stddev=0.005,
                                                                                        dtype=tf.float32))
        biases = tf.get_variable("biases", shape=[n_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local7, weights),biases, name="softmax_linear")#----..,,,,!!!
    return softmax_linear

def losses(logits,labels):
    with  tf.variable_scope("loss")  as scope:

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,
                                                                       name="xebtropy_per_example")
        loss = tf.reduce_mean(cross_entropy,name="loss")

        tf.summary.scalar(scope.name+"loss",loss)
    return loss

def trainning(loss,learning_rate):
    with tf.name_scope("optimizer"):
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0,name="global_step",trainable=False)
        train_op = optimizer.minimize(loss,global_step=global_step)

    return train_op

def evaluation(logists,lables):
    with tf.variable_scope("accurancy") as  scope:

        correct = tf.nn.in_top_k(logists,lables,1)
        correct = tf.cast(correct,tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+"/accuracy",accuracy)
    return accuracy



import  tensorflow as tf
import  numpy as np
from PIL import Image
import input_data

image_W,image_H=288,392
data_path = "./data/"


dic={0:"松软沙砾多",1:"松软沙砾少",2:"压实土壤多",3:"压实土壤少",4:"凹",5:"岩"}
#x=[]

image_list,_=input_data.get_file("./data/ALLall_shuffle_data_lable.txt")

print(len(image_list))
image_list = tf.cast(image_list,tf.string)
image_que = tf.train.slice_input_producer([image_list])
im = tf.read_file(image_que[0])
im = tf.image.decode_png(im,channels=3)
im = tf.image.resize_images(im,(image_W,image_H))
image_batch,name_batch = tf.train.batch([im,image_que[0]], batch_size=16,num_threads=8,capacity=32)
name_batch = tf.reshape(name_batch, [16])

image_batch = tf.cast(image_batch, tf.float32)





saver = tf.train.import_meta_graph('./data/logs/train3four/model.ckpt-2599.meta')
graph = tf.get_default_graph()
input_x = graph.get_tensor_by_name("x:0")
softmax_linear = graph.get_tensor_by_name("softmax_linear/softmax_linear:0")

Kind  = tf.argmax(softmax_linear,1)
# s=0.0
All=[]
with tf.Session() as sess:
    path = tf.train.latest_checkpoint('./data/logs/train3four/')
    saver.restore(sess, path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    try:
        for i in range(301):
            print(i)
            if coord.should_stop():
                break
            image_b,name_b=sess.run([image_batch,name_batch])
            #name_b = [ijk.decode() for ijk in name_b]
            #print(image_b.shape,name_b.shape)
            ans = sess.run(Kind,feed_dict={input_x:image_b})
            for Nkind,na in zip(ans,name_b):
                All.append(na.decode()+" "+str(Nkind))
    except tf.errors.OutOfRangeError:
        print("Stop")
    finally:
        coord.request_stop()
        coord.join(threads)
All = All[:4801]
print(len(All))
with open("./data/ModelLdeep.txt",'w') as f:
    for i in All:
     f.write(str(i)+'\n')





import  tensorflow as tf
import  numpy as np
from PIL import Image
import input_data

image_W,image_H=288,392
data_path = "./data/"


dic={0:"松软沙砾多",1:"松软沙砾少",2:"压实土壤多",3:"压实土壤少",4:"凹",5:"岩"}
#x=[]

image_lsit,label_list = input_data.get_file("./data/VVvv_shuffle_data_lable.txt")
print(len(image_lsit),len(label_list))

#print(x)
#
# x=['./data/5/B336-3.png','./data/0/B100-2.png','./data/3/B558-1.png','./data/3/B522-7.png','./data/3/B176-6.png',
# './data/0/B586-6.png','./data/0/B600-1.png','./data/1/B212-5.png','./data/5/B336-3.png','./data/5/B612-1.png',
# './data/2/B134-8.png','./data/0/B389-9.png','./data/0/B109-9.png','./data/2/B53-7.png','./data/0/B100-2.png',
# './data/4/B235-7.png']

# xx=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# image,lable = input_data.get_batch(x,xx,image_W,image_H,16,20)
vail_batch,vail_label = input_data.get_batch(image_lsit,label_list,288,392,16,32)

saver = tf.train.import_meta_graph('./data/logs/trainfour/model.ckpt-1299.meta')
graph = tf.get_default_graph()
input_x = graph.get_tensor_by_name("x:0")
y = graph.get_tensor_by_name("y:0")
softmax_linear = graph.get_tensor_by_name("softmax_linear/softmax_linear:0")

acc = tf.nn.in_top_k(softmax_linear,y,1)
acc = tf.cast(acc,tf.float16)
acc = tf.reduce_mean(acc)
s=0.0

with tf.Session() as sess:
    path = tf.train.latest_checkpoint('./data/logs/trainfour/')
    saver.restore(sess, path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    try:
        for i in range(171):
            if coord.should_stop():
                break
            image_batch,label_batch=sess.run([vail_batch,vail_label])
            ans = sess.run(acc,feed_dict={input_x:image_batch,y:label_batch})
            s+=ans
            print(i,ans)
    except tf.errors.OutOfRangeError:
        print("Stop")
    finally:
        coord.request_stop()
        coord.join(threads)

print(s/171)
# ii=0
# js=0
# #num = len(x)
# num = len(label_list)
# print(num)


'''

for imaaa,label in zip(image_lsit,label_list):
    ii+=1
    print(ii)
    #js+=1
    #pat=str(imaaa)
    #print(imaaa)
    image = tf.read_file(imaaa)
    image = tf.image.decode_png(image,channels=3)
    image = tf.image.resize_images(image,(image_W,image_H))



    with tf.Session() as sess:
        path= tf.train.latest_checkpoint('./data/logs/train/')
        saver.restore(sess, path)

        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        a = sess.run(image)

        c = np.array(np.zeros(shape=[16,image_W,image_H,3],dtype=np.float))

        for batch in range(0,16):
            c[batch]=a
        #print(c.shape)
        #for k in range(8):
        # b = c[0]
        # #print(b.shape)
        #
        # b = b.reshape(image_W, image_H, 3)
        # b = Image.fromarray(b.astype(np.uint8))
        # #b.show()

        Nkind=sess.run(Kind,feed_dict={input_x:c})
        if Nkind[0]==label:
            js+=1

print(js/num)


        # im = Image.open(imaaa)
        # #im.show()
        # im.save(data_path+"V"+str(Nkind[0])+"/"+imaaa[10:])
        # print("第"+str(js)+"张图片",imaaa[10:],"还剩"+str(num-js)+"张图片")
        #print(dic[Nkind[0]])

'''



import model
import input_data
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import model2


N_CLASSES =4
IMG_W = 288
IMG_H = 392
TRAIN_BATCH_SIZE = 16

VAL_BATCH_SIZE = 16
CAPACITY = 32
MAX_STEP = 1300
learning_rate = 0.0001
def run_training():
    lable_dir = "./data/XXxx_train_shuffle_data_lable.txt"

    logs_train_dir="./data/logs/trainstdfour"
    val_dir="./data/XXxx_vail_shuffle_data_lable.txt"
    val_labelfile=""
    logs_val_dir="./data/logs/valstdfour"
    train_list,trainable_list = input_data.get_file(lable_dir=lable_dir)
    val_list,vallabel_list = input_data.get_file(val_dir)

    train_batch,train_lable_batch = input_data.get_batch(train_list,trainable_list,
                                                         IMG_W,IMG_H,TRAIN_BATCH_SIZE,CAPACITY)
    val_batch,val_label_batch = input_data.get_batch(val_list,vallabel_list,
                                                     IMG_W,IMG_H,VAL_BATCH_SIZE,CAPACITY)
    #logits = model2.inference(train_batch, TRAIN_BATCH_SIZE, N_CLASSES)
    # loss = model2.losses(logits, train_lable_batch)
    # train_op = model2.trainning(loss, learning_rate
    # acc = model2.evaluation(logits, train_lable_batch)

    x =tf.placeholder(tf.float32,shape=[TRAIN_BATCH_SIZE,IMG_W,IMG_H,3],name="x")
    y = tf.placeholder(tf.int32,shape=[TRAIN_BATCH_SIZE],name="y")

    logits = model.inference(x,TRAIN_BATCH_SIZE,N_CLASSES)
    loss = model.losses(logits,y)
    train_op = model.trainning(loss,learning_rate)
    acc = model.evaluation(logits,y)

    with tf.Session() as sess:
      saver = tf.train.Saver()
      sess.run(tf.global_variables_initializer())
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess,coord=coord)
      summary_op = tf.summary.merge_all()

      train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
      val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
      try:
         for step in range(MAX_STEP):
          # if not coord.should_stop():
          #     tra_images,tra_labels = sess.run([train_batch,train_lable_batch])
          #     print(tra_labels)
          #     print(tra_images.shape)
          #
          #     for i in tra_images:
          #             im=i.reshape(IMG_W,IMG_H,3)
          #
          #             im = Image.fromarray(im.astype(np.uint8))
          #             im.show()
              #_,tra_loss,tra_acc= sess.run([train_op,loss,acc],feed_dict={x_train:[1],y_train:[2]})
          tra_images, tra_labels = sess.run([train_batch, train_lable_batch])
          _, tra_loss, tra_acc = sess.run([train_op, loss, acc],feed_dict={x:tra_images,y:tra_labels})
          print(step, tra_loss, tra_acc)
          if step %50 ==0:
              print(step,tra_loss,tra_acc)
              summary_str = sess.run(summary_op,feed_dict={x:tra_images,y:tra_labels})
              train_writer.add_summary(summary_str,step)
          if step % 20==0:
              val_imge,val_labels = sess.run([val_batch,val_label_batch])
              val_loss,val_acc = sess.run([loss,acc],feed_dict={x:val_imge,y:val_labels})
              print("val",step,val_loss,val_acc)
              summary_str = sess.run(summary_op,feed_dict={x:val_imge,y:val_labels})
              val_writer.add_summary(summary_str,step)

          if     (step+1)==MAX_STEP:
               checkpoint_path = os.path.join(logs_train_dir,"model.ckpt")
               saver.save(sess,checkpoint_path,global_step=step)
      except tf.errors.OutOfRangeError:
          print("OutOfRangeError")
      finally:
          coord.request_stop()
          coord.join(threads)

if __name__=="__main__":
    run_training()








import tensorflow as tf
import numpy as np
import csv
import os
import vote_input_model

logs_train_dir="./data/logs/Alltrain"
logs_val_dir="./data/logs/Allval"
MAX_STEP=13000


data,label=vote_input_model.getdata()
N=data.shape[0]*0.7
N = int(N)
train_data,train_label=data[:N],label[:N]
vail_data,vail_label = data[N:],label[N:]

train_data= vote_input_model.P(train_data)
#print(train_data[:5])
vail_data=vote_input_model.P(vail_data)
#print(vail_data[:5],vail_data.shape)

que = tf.train.slice_input_producer([train_data,train_label])
data_batch,label_batch = tf.train.batch([que[0],que[1]],batch_size=16,num_threads=8,capacity=64)



x = tf.placeholder(tf.float32,[None,4],name="x")
y = tf.placeholder(tf.int32,[None],name="y")
logits = vote_input_model.inference(x)
loss = vote_input_model.losses(logits,y)
train_op = vote_input_model.training(loss,0.0001)
acc =vote_input_model.evaluation(logits,y)

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord=tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
    try:
        for step in range(MAX_STEP):
          train_d,train_l=sess.run([data_batch,label_batch])
          #print(train_d[:5])
          _,trian_acc,train_loss = sess.run([train_op,acc,loss],feed_dict={x:train_d,y:train_l})
          vail_acc,vail_loss = sess.run([acc,loss],feed_dict={x:vail_data,y:vail_label})
          if step %100==0:
              print(step, "train:", train_loss, trian_acc)
              summary_str = sess.run(summary_op, feed_dict={x: train_d, y: train_l})
              train_writer.add_summary(summary_str, step)

              print("vail:",vail_loss,vail_acc)
              summary_str = sess.run(summary_op, feed_dict={x: vail_data, y: vail_label})
              val_writer.add_summary(summary_str, step)
          if     (step+1)==MAX_STEP:
               checkpoint_path = os.path.join(logs_train_dir,"model.ckpt")
               saver.save(sess,checkpoint_path,global_step=step)
    except tf.errors.OutOfRangeError:
        print("Out of Range")
    finally:
        coord.request_stop()
        coord.join(threads)

import tensorflow as tf
import numpy as np
import csv
import os
logs_train_dir="./data/logs/Alltrain"
logs_val_dir="./data/logs/Allval"
MAX_STEP=13000

def getdata():
    L=[]
    with open('All.csv','r') as f:
        for i in csv.reader(f):
            L.append(i)
    L.remove(L[0])
    L=np.array(L)
    L=L[:,1:]
    L = np.array(L,np.float)
    # print(L)
    data =L[:,:3]
    label = L[:,3]
    # print(data)
    # print(label)
    label=[int(i) for i in label]
    return  data,label

def P(x):
    a = np.array(np.zeros([x.shape[0],4],np.float))
    for i in range(a.shape[0]):
        for j in range(3):
            a[i,int(x[i,j])]+=1/3
    return a



def inference(x):
    with tf.variable_scope("f1") as scope:
        w=tf.get_variable("w1",[4,500],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        b=tf.get_variable("b1",[500],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        s1 = tf.add(tf.matmul(x,w),b,name="s1")
        s1 = tf.nn.relu(s1)


    # with tf.variable_scope("f2") as scope:
    #
    #     w=tf.get_variable("w2",[100,500],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
    #     b=tf.get_variable("b2",[500],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
    #     s2 = tf.add(tf.matmul(s1,w),b,name="s2")
    #     s2 =tf.nn.relu(s2)

    # with tf.variable_scope("f3") as scope:
    #
    #     w=tf.get_variable("w3",[500,200],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
    #     b=tf.get_variable("b3",[200],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
    #     s3 = tf.add(tf.matmul(s2,w),b,name="s3")
    #     s3 = tf.nn.relu(s3)

    with tf.variable_scope("f4") as scope:

        w=tf.get_variable("w4",[500,4],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        b=tf.get_variable("b4",[4],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        s = tf.add(tf.matmul(s1,w),b,name="s4")


    return s

def losses(s,y):
    with tf.variable_scope("loss2") as scope:
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=s,labels=y)
        loss = tf.reduce_mean(cross_entropy)

        tf.summary.scalar(scope.name + "loss", loss)
    return loss
def training(loss,learning_rate):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op
def evaluation(s,y):
    with tf.variable_scope("accurancy") as  scope:
        y = tf.cast(y,tf.int32)
        correct = tf.nn.in_top_k(s, y, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + "acc", accuracy)
return accuracy




import csv
import numpy as np
import tensorflow as tf
#不用科学计数法表示
np.set_printoptions(suppress=True)

def toInt(x):
    x = np.mat(x)
    n, k = x.shape

    L = np.ones(shape=(n,k))
    for i in range(n):
       for j in range(k):
        L[i,j]= int(x[i,j])
    return L

def normalizing(x):
    n, k = x.shape
    for i in range(n):
       for j in range(k):
        if x[i,j]>=1:
            x[i,j]=1
    return x


with open('train.csv') as aa:
  lst=[]
  for line in csv.reader(aa):
       lst.append(line)
  lst.remove(lst[0])
  lst = np.array(lst)
  lable = lst[:, 0]
  data = lst[:, 1:]
  lable = np.mat(lable)
  lable =toInt(lable)
  data = toInt(data)
  #print(lable,data)
  data = normalizing(data)


#print(data.shape,lable.shape)
#(42000, 784) (1, 42000)
#得到了data和lable
  lable = lable.T
  n, k = lable.shape
  print(n,k)
  L = np.zeros(shape=(n, 10))
  #L = np.mat(L)
  for i in range(n):
     j = int(lable[i,0])
     L[i, j] = 1
  lable = L

print(data.shape,lable.shape)


#-------------------------------------

x_train, y_train, x_test, y_test = data[0:40000, :], lable[0:40000, :], data[40001:, :], lable[40001:, :]



x = tf.placeholder(tf.float32, shape=(None, 784),name='x')
y = tf.placeholder(tf.float32,shape=(None, 10), name='y')

def Dnn(layer1,layer2,solve):
    n=int(layer1.shape[1])
    #初始化----搞清楚
    stddev = 2 / np.sqrt(n)
    init = tf.truncated_normal(shape=(n, layer2), stddev=stddev)
    w = tf.Variable(initial_value=init, name='w')
    b = tf.zeros(layer2,name='b')
    z = tf.matmul(layer1,w)+b
    if solve == 'Relu':
       return (tf.nn.relu(z))

    else:
       return (z)


hidden1 = Dnn(x,200,'Relu')
hidden2 = Dnn(hidden1,200,'Relu')
hidden3 = Dnn(hidden2,100,'Relu')
hidden4= Dnn(hidden3,10,'softmax')
y_hat = tf.nn.softmax(hidden4)

#交叉熵训练
rat = 0.01
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_hat), reduction_indices=1))
train_op = tf.train.GradientDescentOptimizer(rat).minimize(loss)

t =tf.argmax(y_hat, 1,name='t'),
#准确率
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), t), tf.float32), name='acc')
# acc = tf.reduce_mean(tf.cast(tf.equal(y, tf.cast(tf.argmax(y_hat, 1),tf.float32)), tf.float32))

# n = x.shape[1]
# print(n)

#固定
init = tf.global_variables_initializer()
ech=500
save = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    saver = tf.train.import_meta_graph('./modellzw/mnist.meta')
    saver.restore(sess, save_path='./modellzw/mnist')

    graph = tf.get_default_graph()
    for i in range(ech):
        sess.run(train_op, feed_dict={x: x_train, y: y_train})
        #print(y)
        acctrain = acc.eval(feed_dict={x: x_train, y: y_train})

        #accvalid = acc.eval(feed_dict={x: mnis.validation.images, y: mnis.validation.labels})
        acctest = acc.eval(feed_dict={x: x_test, y: y_test})
        if i%10==0:
         print(i+1, acctrain, acctest)
         save.save(sess,save_path='./modellzw/mnist2')



import torch.nn as nn
import torch.nn.functional as F
import math
from ResNestbackbone import resnest50_fast_4s2x40d

from torchsummary import summary
import torch
class DeepwiseConv(nn.Module):
    def __init__(self,inchanel,outchanel,k,stride,dilation):
        super(DeepwiseConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inchanel,
                               out_channels=inchanel,
                               groups=inchanel,
                               kernel_size=k,
                               stride=stride,
                               dilation=dilation,
                               padding=(k // 2) * dilation)
        self.conv2 = nn.Conv2d(in_channels=inchanel, out_channels=outchanel, kernel_size=1, padding=0)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

    def forward(self, x):
        return self.conv2(self.conv1(x))
class ASPP(nn.Module):
    def __init__(self,in_c,out_c,rate):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_c,out_c,1,1,0),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            DeepwiseConv(in_c,out_c,3,1,dilation=6*rate),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            DeepwiseConv(in_c, out_c, 3, 1, dilation=12* rate),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            DeepwiseConv(in_c, out_c, 3, 1, dilation=18 * rate),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.branch5 =nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c,out_c,1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.cat = nn.Sequential(
            nn.Conv2d(out_c*5,out_c,1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        size = x.size()[2:]
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        x5 =F.interpolate(x5,size,mode='bilinear',align_corners=True)
        out = self.cat(torch.cat([x1,x2,x3,x4,x5],dim=1))
        return out



class Deeplabv3P_resnet50(nn.Module):
    def __init__(self,n_class=21):
        super(Deeplabv3P_resnet50, self).__init__()
        self.backbone = resnest50_fast_4s2x40d(num_classes=n_class)
        self.ASPP = ASPP(in_c=2048, out_c=256, rate=2)
        self.low_level_conv1 = nn.Sequential(
            nn.Conv2d(256,48,1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.last_conv = nn.Sequential(
            DeepwiseConv(304,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DeepwiseConv(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_class, 1)
        )

     #TODO 模型加载初始化应该这这里哦

    def forward(self,x):
        size = x.size()[2:]
        out,low =self.backbone(x)
        out = self.ASPP(out)
        out = F.interpolate(out,size=(math.ceil(size[0]/4),
                                   math.ceil(size[1]/4)),mode='bilinear',align_corners=True)

        low = self.low_level_conv1(low)
        out = torch.cat((out,low),dim=1)
        out = self.last_conv(out)
        out = F.interpolate(out,size=size,mode='bilinear',align_corners=True)
        #(out.size())
        return out


# net = Deeplabv3P_resnet50(n_class=2).cuda()
# # d = net.state_dict()
# # print(d.keys())
# summary(net,(3,512,512),device='cuda')
#
# x = torch.randn([2,3,224,224]).cuda()
# net.eval()
# with torch.no_grad():
#     out = net(x)
# print(out.size())
#
# m = [net.backbone]
# print(m[0])
# for i in range(len(m)):
#     for k in m[i].parameters():
#         print(k)


import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from torchsummary import summary
import torch
class DeepwiseConv(nn.Module):
    def __init__(self,inchanel,outchanel,k,stride,dilation):
        super(DeepwiseConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inchanel,
                               out_channels=inchanel,
                               groups=inchanel,
                               kernel_size=k,
                               stride=stride,
                               dilation=dilation,
                               padding=(k // 2) * dilation)
        self.conv2 = nn.Conv2d(in_channels=inchanel, out_channels=outchanel, kernel_size=1, padding=0)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

    def forward(self, x):
        return self.conv2(self.conv1(x))
class ASPP(nn.Module):
    def __init__(self,in_c,out_c,rate):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_c,out_c,1,1,0),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            DeepwiseConv(in_c,out_c,3,1,dilation=6*rate),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            DeepwiseConv(in_c, out_c, 3, 1, dilation=12* rate),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            DeepwiseConv(in_c, out_c, 3, 1, dilation=18 * rate),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.branch5 =nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c,out_c,1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.cat = nn.Sequential(
            nn.Conv2d(out_c*5,out_c,1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.size()[2:]
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        x5 =F.interpolate(x5,size,mode='bilinear',align_corners=True)
        out = self.cat(torch.cat([x1,x2,x3,x4,x5],dim=1))
        return out

class Bottleneck(nn.Module):
    ex = 4
    def __init__(self,
                in_c,
                out_c,
                stride=1,
                d=1,
                g=1,
                downsample = None
                ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_c,out_c,1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

        #self.conv2 = nn.Conv2d(out_c,out_c,3,stride,padding=1*d,dilation=d,groups=g)
        self.conv2 = DeepwiseConv(out_c,out_c,3,stride,d)
        self.bn2 =  nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(out_c,out_c*self.ex, 1)
        self.bn3 = nn.BatchNorm2d(out_c*self.ex)
        self.downsample = downsample

    def forward(self, x):
        r = x
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.relu(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))


        if self.downsample is not None:
            r = self.downsample(x)

        out = out+r
        out = self.relu(out)
        return out


class ResNet_Deeplabv3p_backbone(nn.Module):
    def __init__(self,block,n_block,stride,d=None):
        super(ResNet_Deeplabv3p_backbone, self).__init__()
        self.in_c = 64
        self.conv1 = nn.Conv2d(3,64,7,2,3)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block, 64, n_block[0])

        self.layer2 = self._make_layer(block,128, n_block[1],stride=stride[0])

        self.layer3 = self._make_layer(block,256,n_block[2],stride=stride[1],d=[d[0]]*n_block[2])

        self.layer4 = self._make_layer(block,512,n_block[3],stride=stride[2],d=[d[1]]*n_block[3])


        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
        self._load_pretrain_model()


    def forward(self,x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        low_level_feature = x
        x = self.layer2(x)

        x = self.layer3(x)

        x= self.layer4(x)
        return x,low_level_feature

    def _make_layer(self,block,out_c,n_block,stride=1,d=None,g=None):
        if d is None:
            d=[1]*n_block
        if g is None:
            g=[1]*n_block
        dwomsample = None
        if stride!=1 or self.in_c!=out_c*block.ex:
            dwomsample = nn.Sequential(
                nn.Conv2d(self.in_c,out_c*block.ex,kernel_size=1,stride=stride,padding=0),
                nn.BatchNorm2d(out_c*block.ex)
            )
        layers = []

        layers.append(block(self.in_c,out_c,stride,d[0],g[0],dwomsample))

        self.in_c = out_c*block.ex
        for i in range(1,n_block):
            layers.append(block(self.in_c,out_c,d=d[i]))
        return nn.Sequential(*layers)

    def _load_pretrain_model(self):
        pretrain_net = model_zoo.load_url("https://download.pytorch.org/models/resnet50-19c8e357.pth")
        net_dict = self.state_dict()
        pretrain_net = {v:k for v,k in pretrain_net.items() if (v in net_dict)}
        net_dict.update(pretrain_net)
        self.load_state_dict(net_dict)

#net = ResNet_Deeplabv3p_backbone(Bottleneck,n_block=[3,4,6,3],stride=[2,1,1],d=[2,2]).cuda()

# from torchsummary import summary
# summary(net,(3,224,224),device='cuda')
# print(net)
#print(list(net.state_dict().keys())[:50])
# import torch.utils.model_zoo as model_zoo
# import torch
# model_urls={ 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'}
# pretrain_net = model_zoo.load_url(model_urls['resnet50'])
# net_dict = net.state_dict()
# pretrain_net = {v:k for v,k in pretrain_net.items() if (v in net_dict) }  #先删减自己在更新模型
# net_dict.update(pretrain_net)
# net.load_state_dict(net_dict)
#print(net)
# aspp = ASPP(3,3,2)
# x  = torch.randn([2,3,55,55])
# out = aspp(x)
# print(out.shape)

class Deeplabv3P_resnet50(nn.Module):
    def __init__(self,n_class=21):
        super(Deeplabv3P_resnet50, self).__init__()
        self.backbone = ResNet_Deeplabv3p_backbone(block=Bottleneck,n_block=[3,4,6,3],stride=[2,1,1],d=[2,2])
        self.ASPP = ASPP(in_c=2048, out_c=256, rate=2)
        self.low_level_conv1 = nn.Sequential(
            nn.Conv2d(256,48,1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.last_conv = nn.Sequential(
            DeepwiseConv(304,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DeepwiseConv(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_class, 1)
        )
    def forward(self,x):
        size = x.size()[2:]
        out,low =self.backbone(x)
        out = self.ASPP(out)
        out = F.interpolate(out,size=(math.ceil(size[0]/4),
                                   math.ceil(size[1]/4)),mode='bilinear',align_corners=True)

        low = self.low_level_conv1(low)
        out = torch.cat((out,low),dim=1)
        out = self.last_conv(out)
        out = F.interpolate(out,size=size,mode='bilinear',align_corners=True)
        #(out.size())
        return out


#net = Deeplabv3P_resnet50(2).cuda()
# summary(net,(3,512,512),device='cuda')
#
# x = torch.randn([2,3,224,224]).cuda()
# net.eval()
# with torch.no_grad():
#     out = net(x)
# print(out.size())
#
# m = [net.backbone]
# print(m[0])
# for i in range(len(m)):
#     for k in m[i].parameters():
#         print(k)


import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from splat_learn import Splat

from torchsummary import summary
import torch
class DeepwiseConv(nn.Module):
    def __init__(self,inchanel,outchanel,k,stride,dilation):
        super(DeepwiseConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inchanel,
                               out_channels=inchanel,
                               groups=inchanel,
                               kernel_size=k,
                               stride=stride,
                               dilation=dilation,
                               padding=(k // 2) * dilation)
        self.conv2 = nn.Conv2d(in_channels=inchanel, out_channels=outchanel, kernel_size=1, padding=0)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

    def forward(self, x):
        return self.conv2(self.conv1(x))
class ASPP(nn.Module):
    def __init__(self,in_c,out_c,rate):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_c,out_c,1,1,0),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            DeepwiseConv(in_c,out_c,3,1,dilation=6*rate),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            DeepwiseConv(in_c, out_c, 3, 1, dilation=12* rate),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            DeepwiseConv(in_c, out_c, 3, 1, dilation=18 * rate),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.branch5 =nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c,out_c,1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.cat = nn.Sequential(
            nn.Conv2d(out_c*5,out_c,1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.size()[2:]
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        x5 =F.interpolate(x5,size,mode='bilinear',align_corners=True)
        out = self.cat(torch.cat([x1,x2,x3,x4,x5],dim=1))
        return out

class Bottleneck(nn.Module):
    ex = 4
    def __init__(self,
                in_c,
                out_c,
                stride=1,
                d=1,
                g=1,
                downsample = None,
                card_k=1,
                rdix=1
                ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_c,out_c,1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

        #self.conv2 = nn.Conv2d(out_c,out_c,3,stride,padding=1*d,dilation=d,groups=g)
        #self.conv2 = DeepwiseConv(out_c,out_c,3,stride,d)
        gw = out_c*card_k
        self.conv2 = Splat(in_c=out_c,out_c=gw,k=3,s=stride,p=d,d=d,rdix=rdix,card_k=card_k)
        self.bn2 =  nn.BatchNorm2d(gw)

        self.conv3 = nn.Conv2d(gw,out_c*self.ex, 1)
        self.bn3 = nn.BatchNorm2d(out_c*self.ex)
        self.downsample = downsample

    def forward(self, x):
        r = x
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.relu(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))


        if self.downsample is not None:
            r = self.downsample(x)

        out = out+r
        out = self.relu(out)
        return out


class ResNet_Deeplabv3p_backbone(nn.Module):
    def __init__(self,block,n_block,stride,card_k,rdix,d=None):
        super(ResNet_Deeplabv3p_backbone, self).__init__()
        self.card_k= card_k
        self.rdix =rdix
        self.in_c = 64
        self.conv1 = nn.Conv2d(3,64,7,2,3)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block, 64, n_block[0])

        self.layer2 = self._make_layer(block,128, n_block[1],stride=stride[0])

        self.layer3 = self._make_layer(block,256,n_block[2],stride=stride[1],d=[d[0]]*n_block[2])

        self.layer4 = self._make_layer(block,512,n_block[3],stride=stride[2],d=[d[1]]*n_block[3])


        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
        self._load_pretrain_model()
        self._load_pretrain_model2()


    def forward(self,x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        low_level_feature = x
        #print(low_level_feature.shape)
        x = self.layer2(x)

        x = self.layer3(x)

        x= self.layer4(x)
        return x,low_level_feature

    def _make_layer(self,block,out_c,n_block,stride=1,d=None,g=None):
        if d is None:
            d=[1]*n_block
        if g is None:
            g=[1]*n_block
        dwomsample = None
        if stride!=1 or self.in_c!=out_c*block.ex:
            dwomsample = nn.Sequential(
                nn.Conv2d(self.in_c,out_c*block.ex,kernel_size=1,stride=stride,padding=0),
                nn.BatchNorm2d(out_c*block.ex)
            )
        layers = []

        layers.append(block(self.in_c,out_c,stride,d[0],g[0],dwomsample,card_k=self.card_k,rdix=self.rdix))

        self.in_c = out_c*block.ex
        for i in range(1,n_block):
            layers.append(block(self.in_c,out_c,d=d[i],card_k=self.card_k,rdix=self.rdix))
        return nn.Sequential(*layers)

    def _load_pretrain_model(self):
        pretrain_net = model_zoo.load_url("https://download.pytorch.org/models/resnet50-19c8e357.pth")
        net_dict = self.state_dict()

        pretrain_net = {v:k for v,k in pretrain_net.items() if (v in net_dict)}

        net_dict.update(pretrain_net)
        self.load_state_dict(net_dict)
    def _load_pretrain_model2(self):
        #pr = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50_fast_4s2x40d', pretrained=True).state_dict()
        pr = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True).state_dict()
        net_dict = self.state_dict()

        pr = {v: k for v, k in pr.items() if (v in net_dict)}
        T = {}
        #print(len(pr))
        for v, k in zip(net_dict.items(), pr.items()):
            x1, y1 = v
            x2, y2 = k

            if y1.shape == y2.shape:
                T[x1] = y1
        #         print(x1, y1.shape, x2, y2.shape)
        # print("KKKKKK")
        # print(len(T))
        net_dict.update(T)
        self.load_state_dict(net_dict)




# net = ResNet_Deeplabv3p_backbone(Bottleneck,n_block=[3,4,6,3],stride=[2,1,1],d=[2,2],rdix=2,card_k=1).cuda()
#
# from torchsummary import summary
# summary(net,(3,224,224),device='cuda')
# print(net)
#print(list(net.state_dict().keys())[:50])
# import torch.utils.model_zoo as model_zoo
# import torch
# model_urls={ 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'}
# pretrain_net = model_zoo.load_url(model_urls['resnet50'])
# net_dict = net.state_dict()
# print(list(pretrain_net.keys())[:5])
# print(list(net_dict.keys())[:5])
# pretrain_net = {v:k for v,k in pretrain_net.items() if (v in net_dict) }  #先删减自己在更新模型
# net_dict.update(pretrain_net)
# net.load_state_dict(net_dict)
#print(net)
# aspp = ASPP(3,3,2)
# x  = torch.randn([2,3,55,55])
# out = aspp(x)
# print(out.shape)

class Deeplabv3P_resnet50(nn.Module):
    def __init__(self,card_k,rdix,n_class=21):
        super(Deeplabv3P_resnet50, self).__init__()
        self.backbone = ResNet_Deeplabv3p_backbone(block=Bottleneck,n_block=[3,4,6,3],
                                                   stride=[2,1,1],d=[2,2],card_k=card_k,rdix=rdix)
        self.ASPP = ASPP(in_c=2048, out_c=256, rate=2)
        self.low_level_conv1 = nn.Sequential(
            nn.Conv2d(256,48,1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.last_conv = nn.Sequential(
            DeepwiseConv(304,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DeepwiseConv(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_class, 1)
        )
    def forward(self,x):
        size = x.size()[2:]
        out,low =self.backbone(x)
        out = self.ASPP(out)
        out = F.interpolate(out,size=(math.ceil(size[0]/4),
                                   math.ceil(size[1]/4)),mode='bilinear',align_corners=True)

        low = self.low_level_conv1(low)
        out = torch.cat((out,low),dim=1)
        out = self.last_conv(out)
        out = F.interpolate(out,size=size,mode='bilinear',align_corners=True)
        #(out.size())
        return out


net = Deeplabv3P_resnet50(card_k=1,rdix=2,n_class=2).cuda()
# d = net.state_dict()
# print(d.keys())
summary(net,(3,512,512),device='cuda')
#
# x = torch.randn([2,3,224,224]).cuda()
# net.eval()
# with torch.no_grad():
#     out = net(x)
# print(out.size())
#
# m = [net.backbone]
# print(m[0])
# for i in range(len(m)):
#     for k in m[i].parameters():
#         print(k)



import math
import torch
import torch.nn as nn

from splat import SplAtConv2d

__all__ = ['ResNet', 'Bottleneck']

class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)

class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False,
                 norm_layer=None, dropblock_prob=0.0, last_gamma=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, rectify=rectified_conv,
                rectify_avg=rectify_avg,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob)

        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """ResNet Variants
    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    # pylint: disable=unused-variable
    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64,
                 num_classes=1000, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 rectified_conv=False, rectify_avg=False,
                 avd=False, avd_first=False,
                 final_drop=0.0, dropblock_prob=0,
                 last_gamma=False, norm_layer=nn.BatchNorm2d):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        super(ResNet, self).__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg

        conv_layer = nn.Conv2d
        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False, **conv_kwargs),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
            )
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False, **conv_kwargs)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        elif dilation==2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=1, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self._lod()
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=2, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))

        return nn.Sequential(*layers)
    def _lod(self):
        pr = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50_fast_4s2x40d', pretrained=True).state_dict()

        net_dict = self.state_dict()
        #print(len(pr))
        pr = {v: k for v, k in pr.items() if (v in net_dict)}
        #print(len(pr))
        net_dict.update(pr)
        self.load_state_dict(net_dict)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        low = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #print(x.shape,low.shape)
        return x,low
def resnest50_fast_4s2x40d(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=4, groups=2, bottleneck_width=40,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    return model
# net = resnest50_fast_4s2x40d(num_classes=2).cuda()
# from torchsummary import summary
# summary(net,(3,512,512),device='cuda')
# # print(net)


import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair

__all__ = ['SplAtConv2d']

class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob

        self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)

        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel//self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel//self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x



#from model.Deeplab3p_splat_std import Deeplabv3P_resnet50
from  model_city.Deeplab3p_splat_std import Deeplabv3P_resnet50
import torch
import pandas as pd
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from data.rle import rle_decode,rle_encode
model_path = "/home/lfz/lzw/Project/city/log_resnest50_fast_Data_zq/cityNet7.pth.tar"
#model_path = "/home/lfz/lzw/Project/city/log_resnest50_fast_4s2x40d/cityNet7.pth.tar"
net = Deeplabv3P_resnet50(2)
#from model_city.test import  get_net
#net = get_net()

net.eval()
print(1)
net = net.cuda(device="cuda:0")
model_p = torch.load(model_path,map_location="cuda:0")["state_dict"]
net_p = net.state_dict()
model_p = {v.replace("module.",""):k for v,k in model_p.items()}

print(list(model_p.keys())[:2])
print(list(net_p)[:2])

net.load_state_dict(model_p)

path = "/home/lfz/lzw/tcdata/test_a_samplesubmit.csv"
s_data = pd.read_csv(path,sep='\t',names=["name","mask"])
im = s_data["name"].values
impath = "/home/lfz/lzw/tcdata/test_a/test_a/"

sub_im=[]
sub_label = []

j=0
for i in im:
    j+=1
    print(len(im),':',j)
    image = cv2.imread(os.path.join(impath,i))
    ori_image = image.copy()
    image = np.transpose(image, (2, 0, 1))
    image = image[np.newaxis, :].astype(np.float32)
    image = torch.from_numpy(image).cuda()
    pre = net(image)
    pre = torch.softmax(pre, dim=1)
    # 将channel取得的最大的响应作为标签
    pre = torch.argmax(pre, dim=1)
    # squeeze 将某些维度上的1 去掉
    pre = torch.squeeze(pre)

    pre = pre.detach().cpu().numpy()


    # plt.subplot(1,2,1)
    # plt.imshow(ori_image)
    # plt.subplot(1,2,2)
    # plt.imshow(pre)
    # plt.show()
    # print(pre.shape)

    sub_im.append(i)
    y =rle_encode(pre)
    sub_label.append(y)

sub_path="/home/lfz/lzw/Project/city/log_resnest50_fast_Data_zq/test_resnest50_fast_tversky_7.csv"
sub_data = pd.DataFrame({"name": sub_im, "mask": sub_label})
sub_data.to_csv(sub_path, sep='\t', header=False, index=False)
#print(sub_im,sub_label)


from tqdm import tqdm
import torch
import os
import shutil
from metric import compute_iou
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
#from model.Deeplabv3p import Deeplabv3P_resnet50
from Deeplab3p_splat_std import Deeplabv3P_resnet50
from lossk2 import MySoftmaxCrossEntropyLoss,DiceLoss
from data.get_data import TCDateset,ImageAug, DeformAug,ScaleAug, CutOut,ToTensor,Affine,Fliplr,Flipud
from data.wash_data import wash_d
from test_Tversky import TverskyLoss
import matplotlib.pyplot as plt
import numpy as np


# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

#device_list = [2, 6]
device_list = ['cuda:0']

def train_epoch(net, epoch, dataLoader, optimizer, trainF):
    #model 转化成训练的状态
    net.train()
    total_mask_loss = 0.0

    #这里是一个dataloader
    dataprocess = tqdm(dataLoader)
    for batch_item in dataprocess:

        #取出来batch_item中的value
        image, mask = batch_item['image'], batch_item['mask']
        # print(image.size(),mask.size())
        # x = image[0,:,:,:]
        # y = mask[0,:,:]
        # y = y.squeeze(dim=0)
        #
        # x= x.squeeze(dim=0)
        # plt.subplot(1,3,1)
        # plt.imshow(x.permute(1,2,0)/255.0)
        # print(x.shape)
        # #print(x)
        # plt.subplot(1,3,2)
        # plt.imshow(x.permute(1, 2, 0) /255.0)
        # print(x.shape)
        # print(y)
        # plt.subplot(1,3,3)
        # plt.imshow(y)
        # plt.show()
        #检测环境中是否存在cuda
        if torch.cuda.is_available():
            image, mask = image.cuda(device=device_list[0]), mask.cuda(device=device_list[0])
        #print(image.shape )
        #optimizer.zero将每个parameter的梯度清0

        optimizer.zero_grad()
        #输出预测的mask
        out = net(image)

        #计算交叉熵loss
       # print("-----------------",out.size(),mask.size())
        BC = MySoftmaxCrossEntropyLoss(nbclasses=2).cuda()
        Dice = DiceLoss().cuda()
        Tversky = TverskyLoss(alpha=0.5).cuda()
        # mask_loss = BC(out, mask) + Dice(out, mask)
        mask_loss = BC(out, mask) + Dice(out, mask)
        total_mask_loss += mask_loss.item()


        #进行后向计算
        mask_loss.backward()

        #optimizer进行更新
        optimizer.step()


        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss.item()))
    #记录数据迭代了多少次
    trainF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(dataLoader)))
    trainF.flush()


def test(net, epoch, dataLoader, testF):
    #将model转化成了eval
    net.eval()
    total_mask_loss = 0.0
    dataprocess = tqdm(dataLoader)

    result = {"TP": {i:0 for i in range(8)}, "TA":{i:0 for i in range(8)}}
    for batch_item in dataprocess:
        image, mask = batch_item['image'], batch_item['mask']
        if torch.cuda.is_available():
            image, mask = image.cuda(device=device_list[0]), mask.cuda(device=device_list[0])
        out = net(image)
        BC = MySoftmaxCrossEntropyLoss(nbclasses=2).cuda()
        Dice = DiceLoss().cuda()
        #Tversky = TverskyLoss(alpha=0.5).cuda()
        #mask_loss = BC(out, mask) + Dice(out, mask)
        mask_loss =BC(out,mask)+ Dice(out,mask)

        #detach（）截断梯度的作用，可以不截断，查一下用法
        total_mask_loss += mask_loss.detach().item()
        pred = torch.argmax(F.softmax(out, dim=1), dim=1)

        #计算iou
        result = compute_iou(pred, mask, result)
        dataprocess.set_description_str("epoch:{}".format(epoch))

        dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss))
    testF.write("Epoch:{} \n".format(epoch))
    #求出每一个类别的iou
    ss =0.0
    for i in range(2):
        result_string = "{}: {:.4f} \n".format(i, result["TP"][i]/result["TA"][i])
        ss+= result["TP"][i]/result["TA"][i]
        print(result_string)


        #写入log文件
        testF.write(result_string)
    print(ss/2)
    ss = str(ss/2)+'\n'
    testF.write(ss)
    testF.write("Epoch:{}, mask loss is {:.4f} \n".format(epoch, total_mask_loss / len(dataLoader)))
    testF.flush()





    #设置model parameters

log_path = "/home/lfz/lzw/Project/city/log_resnest50_fast_Data_zq"

#查看路径是否存在
if os.path.exists(log_path):
    #如果存在的话，全部删掉
    shutil.rmtree(log_path)
#建立一个新的文件件
os.makedirs(log_path, exist_ok=True)

#打开文件夹，在这两个文件内记录
trainF = open(os.path.join(log_path, "train.csv"), 'w')
testF = open(os.path.join(log_path, "test.csv"), 'w')

#set up dataset
# 'pin_memory'意味着生成的Tensor数据最开始是属于内存中的索页，这样的话转到GPU的显存就会很快
kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

temp_train_newcsv = "/home/lfz/lzw/tcdata/train_mask.csv/temp_train_new_train_mask.csv"
temp_test_newcsv =  "/home/lfz/lzw/tcdata/train_mask.csv/temp_test_new_train_mask.csv"
train_newcsv = "/home/lfz/lzw/tcdata/train_mask.csv/train_new_train_mask.csv"
path = "/home/lfz/lzw/tcdata/train/train"
#set up training dataset
train_dataset = TCDateset(train_newcsv,path,transform=transforms.Compose([ ToTensor()],))

#set up training dataset 的dataloader
train_data_batch = DataLoader(train_dataset, batch_size=3*len(device_list), shuffle=True, drop_last=True, **kwargs)
test_newcsv = "/home/lfz/lzw/tcdata/train_mask.csv/test_new_train_mask.csv"

#set ip validation dataset
val_dataset = TCDateset(test_newcsv,path,transform=transforms.Compose([ToTensor()]))

#set up validation dataset's dataloader
val_data_batch = DataLoader(val_dataset, batch_size=2*len(device_list), shuffle=False, drop_last=False, **kwargs)

#test
# device_list = ['cuda:0']
# train_dataset = LaneDataset("train.csv", transform=transforms.Compose([ImageAug(), DeformAug(),
#                                                                            ScaleAug(), CutOut(32, 0.5), ToTensor()]))
# train_data_batch = DataLoader(train_dataset, batch_size=4*len(device_list), shuffle=True, drop_last=True)
# for batch in train_data_batch:
#     im,mask = batch['image'],batch['mask']
#     im,mask =  im.cuda(),mask.cuda()
#     print(im.size(),mask.size())



#build model


#from model_city.test import  get_net
net = Deeplabv3P_resnet50(n_class=2)
#net = get_net()


#检测一下环境中是否存在GPU，存在的话就转化成cuda的格式
if torch.cuda.is_available():
    net = net.cuda(device=device_list[0])

    net = torch.nn.DataParallel(net, device_ids=device_list)

#config the optimizer
# optimizer = torch.optim.SGD(net.parameters(), lr=lane_config.BASE_LR,
#                              momentum=0.9, weight_decay=lane_config.WEIGHT_DECAY)

#查一下weight_decay的作用
#optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
optimizer = torch.optim.AdamW(net.parameters(), lr=1.0e-4, weight_decay=1.0e-3)

#Training and test
def adjust_lr(optimizer, epoch):

    #多机多卡上的 trick：warming up
    if epoch == 0:
        lr = 1e-3
    elif epoch == 2:
        lr = 0.0006
    elif epoch == 4:
        lr = 0.0003
    elif epoch == 6:
        lr = 0.0001

    else:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(12):
    wash_d()
    #adjust_lr(optimizer, epoch)
    #在train_epoch中
    train_epoch(net, epoch, train_data_batch, optimizer, trainF)

    test(net, epoch, val_data_batch, testF)

    if epoch % 1 == 0:
        torch.save({'state_dict': net.state_dict()}, os.path.join(os.getcwd(), log_path, "cityNet{}.pth.tar".format(epoch)))
trainF.close()
testF.close()


torch.save({'state_dict': net.state_dict()}, os.path.join(os.getcwd(), log_path, "finalNet.pth.tar"))



from losskk import BinaryDiceLoss,make_one_hot,DiceLoss
import torch.nn as nn
import torch
import torch.nn.functional as F

# D = DiceLoss(p=1).cuda()
# x = torch.randn([2,3,5,5]).cuda()
# y = torch.empty([2,5,5],dtype=torch.long).random_(3).cuda()
#
# out = D(x,y)
# print(out)

class BinaryTverskyLoss(nn.Module):
    def __init__(self,alpha=0.5,smooth=1e-4):
        super(BinaryTverskyLoss, self).__init__()
        self.alpha=alpha
        self.smooth = smooth
    def forward(self,x,y):
        batch_size = x.shape[0]
        loss = 0.0
        alpha = self.alpha
        beta = 1-alpha
        for i in range(batch_size):
            tp =(x[i]*y[i]).sum()
            #print(i,"btp",tp)
            fp = (x[i]*(1-y[i])).sum()

            fn = ((1-x[i])*y[i]).sum()
            tversky = (tp+self.smooth)/(alpha*fp+tp+beta*fn+self.smooth)
            loss +=1-tversky
        return loss/batch_size
# DIC = BinaryTverskyLoss().cuda()
# out2 = DIC(x,y)
# print(out2)
class TverskyLoss(nn.Module):
    def __init__(self,**kwargs):
        super(TverskyLoss, self).__init__()
        self.kwargs = kwargs
    def forward(self, predict, target):
        target = torch.unsqueeze(target,1)
        target=make_one_hot(target, predict.shape[1])
        assert predict.shape == target.shape, 'predict & target shape do not match'
        binarytversky = BinaryTverskyLoss(**self.kwargs)
        #binarytversky = BinaryDiceLoss(p=1)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):

                tversky = binarytversky(predict[:, i], target[:, i])

                #print(i,tversky)
                total_loss += tversky

        return total_loss/target.shape[1]

#
# T = TverskyLoss(alpha=0.5).cuda()
#
# out2 = T(x,y)
# print(out2)



import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MySoftmaxCrossEntropyLoss(nn.Module):

    def __init__(self, nbclasses):
        super(MySoftmaxCrossEntropyLoss, self).__init__()
        self.nbclasses = nbclasses

    def forward(self, inputs, target):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, self.nbclasses)  # N,H*W,C => N*H*W,C
        target = target.view(-1)
        return nn.CrossEntropyLoss(reduction="mean")(inputs, target)


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.alpha = torch.tensor([alpha, 1 - alpha])
        self.size_average = size_average

    def forward(self, inputs, target):
        if inputs.dim() > 2:
            inputs = inputs
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(inputs,dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, target.view(-1))
            logpt = logpt * at
        # mask = mask.view(-1)
        loss = -1 * (1 - pt) ** self.gamma * logpt #* mask
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    result = result.cuda()

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        num = 2*torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        #print("num",num)
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        # print("num",num)
        # print("den",den)
        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index


    def forward(self, predict, target):
        target = torch.unsqueeze(target,1)
        target=make_one_hot(target, predict.shape[1])
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
               # print(i,dice_loss)

                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]
# x = torch.empty([2,4],dtype=torch.long).random_(8)
# print(make_one_hot(x,10).shape)



from torch.utils.data import Dataset
import torch
import pandas as pd
import cv2
import os
from data.rle import rle_decode,rle_encode
import  numpy as np
from imgaug import augmenters as iaa
import random

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
class TCDateset(Dataset):
    def __init__(self,csvpath,path,transform=None):
        super(TCDateset, self).__init__()
        self.path = path
        self.data = pd.read_csv(csvpath,header=None, names=['name', 'mask'])
        self.image = self.data["name"].values[1:]
        self.label = self.data["mask"].values[1:]
        self.transform=transform

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, item):
        #print(item)
        image = cv2.imread(os.path.join(self.path,self.image[item]))
        # image = np.transpose(image, (2, 0, 1))
        # image = image.astype(np.float32)

        label = rle_decode(self.label[item])
        # label= label.astype(np.long)
        smaple = [image.copy(), label.copy()]
        if self.transform:
           smaple = self.transform(smaple)

        # image, label = smaple
        # return {"image":torch.from_numpy(image),
        #         "mask":torch.from_numpy(label)}
        return smaple
# csvpath = "/home/lfz/lzw/tcdata/train_mask.csv/train_mask.csv"
# newcsv = "/home/lfz/lzw/tcdata/train_mask.csv/new_train_mask.csv"
# path = "/home/lfz/lzw/tcdata/train/train"
# data = pd.read_csv(newcsv, header=None, names=['name', 'mask'])
# image = data["name"].values[1:]
# label = data["mask"].values[1:]
# import numpy
# image = cv2.imread(os.path.join(path,image[2]))
#
# for i in range(len(label)):
#     print(rle_decode(label[i]).shape)
class ImageAug(object):
    def __call__(self, sample):
        image, mask = sample
        if np.random.uniform(0,1) > 0.5:
            seq = iaa.Sequential([iaa.OneOf([
                iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),
                iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3)),
                iaa.GaussianBlur(sigma=(0, 1.0))])])
            image = seq.augment_image(image)
        return image, mask


# deformation augmentation
class DeformAug(object):
    def __call__(self, sample):
        image, mask = sample
        seq = iaa.Sequential([iaa.CropAndPad(percent=(-0.05, 0.1))])
        seg_to = seq.to_deterministic()
        image = seg_to.augment_image(image)
        mask = seg_to.augment_image(mask)
        return image, mask

class Fliplr(object):
    def __call__(self,sample):
        image,mask = sample
        seq = iaa.Sequential(iaa.Fliplr(p=0.5, name=None, random_state=None))
        image = seq.augment_image(image)
        mask = seq.augment_image(mask)
        return image,mask

class Flipud(object):
    def __call__(self,sample):
        image,mask = sample
        seq = iaa.Sequential(iaa.Flipud(p=0.5, name=None,  random_state=None))
        image = seq.augment_image(image)
        mask = seq.augment_image(mask)
        return image,mask

class Affine(object):
    def __call__(self,sample):
        image,mask = sample
        rota = random.uniform(-30.0, 30.0)
        # print(rota)
        seq = iaa.Sequential(iaa.Affine(scale=1.0,
                   translate_percent=None,
                   translate_px=None,
                   rotate=rota,
                   shear=rota,
                   order=0,
                   cval=0,
                   mode='constant',
                   name=None, random_state=None))
        #seq = seq.to_deterministic()
        image = seq.augment_image(image)
        mask = seq.augment_image(mask)
        return image,mask



class ScaleAug(object):
    def __call__(self, sample):
        image, mask = sample
        scale = random.uniform(0.7, 1.5)
        h, w, _ = image.shape
        aug_image = image.copy()
        aug_mask = mask.copy()
        aug_image = cv2.resize(aug_image, (int (scale * w), int (scale * h)))
        aug_mask = cv2.resize(aug_mask, (int (scale * w), int (scale * h)))
        if (scale < 1.0):
            new_h, new_w, _ = aug_image.shape
            pre_h_pad = int((h - new_h) / 2)
            pre_w_pad = int((w - new_w) / 2)
            pad_list = [[pre_h_pad, h - new_h - pre_h_pad], [pre_w_pad, w - new_w - pre_w_pad], [0, 0]]
            aug_image = np.pad(aug_image, pad_list, mode="constant")
            aug_mask = np.pad(aug_mask, pad_list[:2], mode="constant")
        if (scale > 1.0):
            new_h, new_w, _ = aug_image.shape
            pre_h_crop = int ((new_h - h) / 2)
            pre_w_crop = int ((new_w - w) / 2)
            post_h_crop = h + pre_h_crop
            post_w_crop = w + pre_w_crop
            aug_image = aug_image[pre_h_crop:post_h_crop, pre_w_crop:post_w_crop]
            aug_mask = aug_mask[pre_h_crop:post_h_crop, pre_w_crop:post_w_crop]
        return aug_image, aug_mask


class CutOut(object):
    def __init__(self, mask_size, p):
        self.mask_size = mask_size
        self.p = p

    def __call__(self, sample):
        image, mask = sample
        mask_size_half = self.mask_size // 2
        offset = 1 if self.mask_size % 2 == 0 else 0

        h, w = image.shape[:2]
        cxmin, cxmax = mask_size_half, w + offset - mask_size_half
        cymin, cymax = mask_size_half, h + offset - mask_size_half
        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin, ymin = cx - mask_size_half, cy - mask_size_half
        xmax, ymax = xmin + self.mask_size, ymin + self.mask_size
        xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax)
        if np.random.uniform(0, 1) < self.p:
            image[ymin:ymax, xmin:xmax] = (0, 0, 0)
        return image, mask


class ToTensor(object):
    def __call__(self, sample):

        image, mask = sample
        image = np.transpose(image,(2,0,1))
        image = image.astype(np.float32)
        mask = mask.astype(np.long)
        return {'image': torch.from_numpy(image.copy()),
                'mask': torch.from_numpy(mask.copy())}
