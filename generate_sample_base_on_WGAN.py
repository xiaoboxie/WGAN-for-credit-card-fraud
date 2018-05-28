# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:09:20 2018

@author: jxiong
"""

# coding:utf-8
"""
WGAN
- D的最后一层取消sigmoid
- 损失函数取消log
- D的w 取值限制在[-c,c]区间内
- 使用RMSProp或SGD并以较低的学习率进行优化
"""

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.contrib.layers as ly
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import itertools

#########################################################################
feature_dim = 18
steps=20000
batch_size=492
credit_card_path='creditcard.csv'
generate_path='forpaper/生成欺诈数据'
model_save_path='forpaper/check_point'
is_restore = False




tf.reset_default_graph()
def sigmoid(matrix):
    res = 1/(1+np.exp(-matrix))
    return res

def relu(matrix):
    res = np.maximum(matrix, 0)
    return res
    
def discriminator(para):
    para[x] = para[x].astype(np.float32)
    d_output = np.dot(para[x], para[D_PARAMS[0]]) + para[D_PARAMS[1]]
    d_output = relu(d_output)
    d_output2 = np.dot(d_output, para[D_PARAMS[2]]) + para[D_PARAMS[3]]
    d_output2 =relu(d_output2)
    d_output3_ = np.dot(d_output2, para[D_PARAMS[4]]) + para[D_PARAMS[5]]
    d_output3 = relu(d_output3_)
    d_output4_ = np.dot(d_output3, para[D_PARAMS[6]]) + para[D_PARAMS[7]]
    d_output4 = relu(d_output4_)
    d_output5_ = np.dot(d_output4, para[D_PARAMS[8]]) + para[D_PARAMS[9]]
    d_output5 = relu(d_output5_)
    return d_output5, d_output5_

def make_range_table(score):
    amount = []
    lowerbound = -5.0
    while lowerbound<20:  
        count = 0
        for i in score:
            
            
            if lowerbound<=i[0]<lowerbound + 0.1:
                count += 1
        amount.append(count)
        lowerbound += 0.1
    x = list(np.arange(-5,20,0.1))
    y = amount
    plt.plot(x,y)

def dense(inputs, shape, name, bn=False, act_fun=None):
    W = tf.get_variable(name + ".w", initializer=tf.random_normal(shape=shape))
    b = tf.get_variable(name + ".b", initializer=(tf.zeros((1, shape[-1])) + 0.1))
    y = tf.add(tf.matmul(inputs, W), b)

    def batch_normalization(inputs, out_size, name, axes=0):
        mean, var = tf.nn.moments(inputs, axes=[axes])
        scale = tf.get_variable(name=name + ".scale", initializer=tf.ones([out_size]))
        offset = tf.get_variable(name=name + ".shift", initializer=tf.zeros([out_size]))
        epsilon = 0.001
        return tf.nn.batch_normalization(inputs, mean, var, offset, scale, epsilon, name=name + ".bn")

    if bn:
        y = batch_normalization(y, shape[1], name=name + ".bn")
    if act_fun:
        y = act_fun(y)
    return y


def D(inputs, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        l1 = dense(inputs, [feature_dim, 100], name="relu1", act_fun=tf.nn.relu)
        l2 = dense(l1, [100, 100], name="relu2", act_fun=tf.nn.relu)
        l3 = dense(l2, [100, 100], name="relu3", act_fun=tf.nn.relu)
        l4 = dense(l3, [100, 100], name="relu4", act_fun=tf.nn.relu)
        y = dense(l4, [100, 1], name="output")
        return y


def G(inputs, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        l1 = dense(inputs, [100, 100], name="relu1", act_fun=tf.nn.relu)
        l2 = dense(l1, [100, 100], name="relu2", act_fun=tf.nn.relu)
        l3 = dense(l2, [100, 100], name="relu3", act_fun=tf.nn.relu)
        l4 = dense(l3, [100, 100], name="relu4", act_fun=tf.nn.relu)
        y = dense(l4, [100, feature_dim], name="output", bn=True)
        return y

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

import pickle
def pickle_generate_sample(samples, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(samples, f, 0)
    

def generate_samples(steps=steps, batch_size=batch_size, credit_card_path=credit_card_path, generate_path=generate_path,\
                     model_save_path=model_save_path, is_restore = is_restore):
    z = tf.placeholder(tf.float32, [None, 100], name="noise")  # 100
    x = tf.placeholder(tf.float32, [None, feature_dim], name="pos_data")  # 28*28
    neg_data = tf.placeholder(tf.float32, [None, feature_dim], name="neg_data")
    
    real_out = D(x, "D", reuse = False)
    gen = G(z, "G")
    fake_out = D(gen, "D", reuse=True)
    neg_data_out = D(neg_data, "D", reuse=True)
    
    vars = tf.trainable_variables()
    
    D_PARAMS = [var for var in vars if var.name.startswith("D")]
    G_PARAMS = [var for var in vars if var.name.startswith("G")]
    
    d_clip = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in D_PARAMS]
    d_clip = tf.group(*d_clip)  # 限制参数
    
    wd = tf.reduce_mean(real_out) - tf.reduce_mean(fake_out)
    d_loss = tf.reduce_mean(fake_out) - tf.reduce_mean(real_out) + tf.reduce_mean(neg_data_out)
    g_loss = tf.reduce_mean(-fake_out)
    
    d_opt = tf.train.AdamOptimizer(1e-4, 0.5).minimize(
        d_loss,
        global_step=tf.Variable(0),
        var_list=D_PARAMS
    )
    
    g_opt = tf.train.AdamOptimizer(1e-4, 0.5).minimize(
        g_loss,
        global_step=tf.Variable(0),
        var_list=G_PARAMS
    )
    # is_restore = True  # 是否第一次训练(不需要载入模型)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    if is_restore:
        saver = tf.train.Saver()
        # 提取变量
        saver.restore(sess, model_save_path+"/GAN_net.ckpt")
        print("Model restore...")
    
    
    CRITICAL_NUM = 5
    
    dataset = pd.read_csv(credit_card_path)
    
    pitch_size = dataset.shape[0]//5 + 1
    
    train_set = dataset.loc[:pitch_size*4]
    
    droplist = ['Time']
    
    train_set['Hour'] =train_set["Time"].apply(lambda x : divmod(x, 3600)[0]) 
    data_new = train_set.drop(droplist, axis = 1)
    col = ['Amount','Hour']
    from sklearn.preprocessing import StandardScaler # 导入模块
    sc =StandardScaler() # 初始化缩放器
    data_new[col] =sc.fit_transform(data_new[col])#对数据进行标准化
    data_new.head()
    
    pos_set = data_new[data_new['Class']==1]
    neg_set = data_new[data_new['Class']==0]
    neg_set = neg_set.drop('Class', axis=1)
    generate_num = pitch_size*4 - (train_set['Class']==1).shape[0]
    #处理不平衡数据
    X = pos_set.drop('Class', axis=1)
    y = pos_set['Class']
  
    
    for step in range(steps):
        if step < 25 or step % 500 == 0:
            critical_num = 100
        else:
            critical_num = CRITICAL_NUM
        
        for ep in range(critical_num):
            noise = np.random.normal(size=(batch_size, 100))
            
            batch_xs = X.sample(batch_size, replace = True)
            
            _, d_loss_v, _ = sess.run([d_opt, d_loss, d_clip], feed_dict={
                x: batch_xs,
                neg_data: neg_set.sample(batch_size, replace = False),
                z: noise
                })
    
        for ep in range(1):
            noise = np.random.normal(size=(64, 100))
            _, g_loss_v = sess.run([g_opt, g_loss], feed_dict={
                z: noise
            })
        if step%20 == 0:
            print("Step:%d   D-loss:%.4f    G-loss:%.4f" %(step+1, d_loss_v, g_loss_v))
        if step % 1000 == 999:
            batch_xs = X.sample(64, replace = True)#第一个元素是图像，第二个元素是one-hot编码
            noise = np.random.normal(size=(generate_num, 100))
            mpl_v = sess.run(wd, feed_dict={
                x: batch_xs,
                z: noise
            })
            print("##################    Step %d  WD:%.4f ###############" % (step + 1, mpl_v))
            generate = sess.run(gen, feed_dict={
                z: noise
            })
            print(generate)
            
            pickle_generate_sample(generate, generate_path+'/step=%s' %(step+1))
            saver = tf.train.Saver()
            save_path = saver.save(sess, model_save_path+'/GAN_net.ckpt')
            print("Model save in %s" % save_path)
    sess.close()

if __name__ == '__main__':
    generate_samples()
    

    
    
    





    
