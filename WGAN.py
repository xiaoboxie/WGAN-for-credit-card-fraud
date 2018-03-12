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
import pickle

SAMPLE_SIZE = 29
#dataset = pd.read_csv('creditcard.csv')
#X = dataset
#del X['Class']
#
#y = dataset['Class']
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
#param_grid = {'C':[0.01,0.1,1,10,100,1000,], 'penalty': ['l1', 'l2']}
#grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=10)
#grid_search.fit(X_train, y_train)
#
#results = pd.DataFrame(grid_search.cv_results_)
#best = np.argmax(results.mean_test_score.values)
#print('Best parameters: {}'.format(grid_search.best_params_))
#print('Best cross-validation score: {:.5f}'.format(grid_search.best_score_))
#y_pred = 





#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


#def combine(image):
#    assert len(image) == 64#图片长度必须为64,否则报错
#    rows = []
#    for i in range(8):
#        cols = []
#        for j in range(8):
#            index = i * 8 + j
#            img = image[index].reshape(28, 28)
#            cols.append(img)
#        row = np.concatenate(tuple(cols), axis=0)
#        rows.append(row)
#    new_image = np.concatenate(tuple(rows), axis=1)
#    return new_image.astype("uint8")

tf.reset_default_graph()#防止tf.variable_scope()导致的错误

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
#    d_output6_ = np.dot(d_output5, para[D_PARAMS[10]]) + para[D_PARAMS[11]]
#    d_output6 = relu(d_output6_)
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
        l1 = dense(inputs, [SAMPLE_SIZE, 100], name="relu1", act_fun=tf.nn.relu)
        l2 = dense(l1, [100, 100], name="relu2", act_fun=tf.nn.relu)
        l3 = dense(l2, [100, 100], name="relu3", act_fun=tf.nn.relu)
        l4 = dense(l3, [100, 100], name="relu4", act_fun=tf.nn.relu)
        #l5 = dense(l4, [512, 512], name="relu5", act_fun=tf.nn.relu)
        y = dense(l4, [100, 1], name="output")
        return y


def G(inputs, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        l1 = dense(inputs, [100, 100], name="relu1", act_fun=tf.nn.relu)
        l2 = dense(l1, [100, 100], name="relu2", act_fun=tf.nn.relu)
        l3 = dense(l2, [100, 100], name="relu3", act_fun=tf.nn.relu)
        l4 = dense(l3, [100, 100], name="relu4", act_fun=tf.nn.relu)
        #l5 = dense(l4, [512, 512], name="relu5", act_fun=tf.nn.relu)
        y = dense(l4, [100, SAMPLE_SIZE], name="output", bn=True, act_fun=tf.nn.sigmoid)
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

def pickle_load_generate_sample(generate_file):
    with open('C:/Users/jxiong/Desktop/forpaper/生成欺诈数据/%s' %generate_file, 'rb') as f:
        fraud_sample_generated = pickle.load(f)
        fraud_sample_generated = pd.DataFrame(fraud_sample_generated, columns = \
                                              ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',\
                                               'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', \
                                               'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', \
                                               'V28', 'Amount'])
        fraud_sample_generated['Class'] = 1
        return fraud_sample_generated
        
z = tf.placeholder(tf.float32, [None, 100], name="noise")  # 100
x = tf.placeholder(tf.float32, [None, SAMPLE_SIZE], name="image")  # 28*28
fraud_sample = tf.placeholder(tf.float32, [None, SAMPLE_SIZE], name="fraud")

real_out = D(x, "D", reuse = False)
gen = G(z, "G")
fake_out = D(gen, "D", reuse=True)
fraud_out = D(fraud_sample, "D", reuse = True)

vars = tf.trainable_variables()

D_PARAMS = [var for var in vars if var.name.startswith("D")]
G_PARAMS = [var for var in vars if var.name.startswith("G")]

d_clip = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in D_PARAMS]
d_clip = tf.group(*d_clip)  # 限制参数

wd = tf.reduce_mean(real_out) - tf.reduce_mean(fake_out)
d_loss =  - tf.reduce_mean(real_out) + tf.reduce_mean(fraud_out)
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
is_restore = False
# is_restore = True  # 是否第一次训练(不需要载入模型)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if is_restore:
    saver = tf.train.Saver()
    # 提取变量
    saver.restore(sess, "my_net/GAN_net.ckpt")
    print("Model restore...")


CRITICAL_NUM = 5

def train_part(steps, train_set):
    #处理不平衡数据
#    from imblearn.over_sampling import SMOTE
#    X = train_set.drop('Class', axis=1)
#    y = train_set['Class']
#    
#    columns = X.columns
#    sm = SMOTE(random_state=42)
#    X,y = sm.fit_sample(X, y)
#    
#    print('通过SMOTE方法平衡正负样本后')
#    n_sample = y.shape[0]
#    n_pos_sample = y[y == 0].shape[0]
#    n_neg_sample = y[y == 1].shape[0]
#    print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample, n_pos_sample/n_sample, n_neg_sample/n_sample))
#    
#    
#    
#    X = pd.DataFrame(X, columns = columns)
#    X['Class'] = y

    trainset = train_set[train_set['Class']==0]
    #df['Amount_max_fraud'] = 1
    #df.loc[df.Amount <= 2125.87, 'Amount_max_fraud'] = 0
    del trainset['Class']
#    del trainset['Amount']
#    del trainset['Time']
    del trainset['Hour']
    #欺诈部分的训练集
    
    df_fraud = train_set[train_set['Class']==1]
    #df_fraud['Amount_max_fraud'] = 1
    #df_fraud.loc[df_fraud.Amount <= 2125.87, 'Amount_max_fraud'] = 0
    del df_fraud['Class']
#    del df_fraud['Amount']
#    del df_fraud['Time']
    del df_fraud['Hour']
    
    for step in range(steps):
        if step < 25 or step % 500 == 0:
            critical_num = 100
        else:
            critical_num = CRITICAL_NUM
        
        for ep in range(critical_num):
            noise = np.random.normal(size=(492, 100))
            
            batch_xs = trainset.sample(492, replace = True)
            
            _, d_loss_v, _ = sess.run([d_opt, d_loss, d_clip], feed_dict={
                x: batch_xs,
                fraud_sample: df_fraud.sample(492, replace = False),
                z: noise
            })
        final_D_para = sess.run(D_PARAMS)#final_D_parameter
    
        for ep in range(1):
            noise = np.random.normal(size=(64, 100))
            _, g_loss_v = sess.run([g_opt, g_loss], feed_dict={
                z: noise
            })
        if step%500 == 0:
            print("Step:%d   D-loss:%.4f" %(step+1, -d_loss_v))
    #    print("Step:%d   D-loss:%.4f  G-loss:%.4f" % (step + 1, -d_loss_v, g_loss_v))
        if step % 1000 == 999:
            batch_xs = trainset.sample(64, replace = True)#第一个元素是图像，第二个元素是one-hot编码
            # batch_xs = pre(batch_xs)
            noise = np.random.normal(size=(64, 100))
            mpl_v = sess.run(wd, feed_dict={
                x: batch_xs,
                z: noise
            })
            print("##################    Step %d  WD:%.4f ###############" % (step + 1, mpl_v))
            generate = sess.run(gen, feed_dict={
                z: noise
            })
    return final_D_para        
    
    #        generate *= 255
    #        generate = np.clip(generate, 0, 255)
    #        image = combine(generate)
    #        Image.fromarray(image).save("image/Step_%d.jpg" % (step + 1))
    #        saver = tf.train.Saver()
    #        save_path = saver.save(sess, "my_net/GAN_net.ckpt")
    #        print("Model save in %s" % save_path)
    sess.close()

def test_part(final_D_para,test_set, score_threshold=-0.1): 
    #测试集数据生成

    test_result = test_set['Class']
    #fraud_count = test_set['Class'].sum()
    #df_test['Amount_max_fraud'] = 1
    #df_test.loc[df_test.Amount <= 2125.87, 'Amount_max_fraud'] = 0
    del test_set['Class']
#    del test_set['Amount']
#    del test_set['Time']
    del test_set['Hour']
    
    
    test_data = np.array(test_set)
    test_result = np.array(test_result)
    
    para = {}
    for i in range(len(final_D_para)):
        para[D_PARAMS[i]] = final_D_para[i]
    
    para[x] = test_data
    
    prediction,score = discriminator(para)#sess.run([D_output3,D_output3_], feed_dict=para)
    prediction = []
    print(score)    

    for each_prediction in score:
        if each_prediction[0] >= score_threshold:
            prediction.append(0)
        else:
            prediction.append(1)
    
    count = 0
    fraud_count_prediction = 0
    fraud_count_prediction_error = 0
    
    save_fraud_score = []
    save_nonfraud_score = []
    
    for i in range(len(test_result)):
        if test_result[i] == 1:
            save_fraud_score.append(score[i])
        else:
            save_nonfraud_score.append(score[i])
        if test_result[i] == prediction[i]:
            count += 1
            if test_result[i] == 1:
                fraud_count_prediction += 1
        else:
            if test_result[i] == 0:
                fraud_count_prediction_error += 1
                
    #table

    print(classification_report(test_result, prediction))
    
    #Compute confusion matrix
    cnf_matrix = confusion_matrix(test_result, prediction)#生成混淆矩阵
    np.set_printoptions(precision=2)
    print("Recall metric in the testing dataset:",cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))    
    #Plot non-normalized confusion matrix
    class_names = [0,1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes = class_names, title = 'Confusion matrix')
    plt.show()
    
    #绘制ROC曲线
    fpr, tpr, thresholds = roc_curve(test_result, prediction)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Oprating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.5f' %roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1], 'r--')
    plt.xlim([-0.1,1.0])
    plt.ylim([-0.1,1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    accuracy = count/len(test_result)

    print ('the prediction accuracy is %1.4f (full)' %(accuracy*100), end = '')
    print('%')
#    print('there are %s times fraud, and %s times have been predict accurately' %(fraud_count, fraud_count_prediction))
#    if fraud_count_prediction_error:
#        print('and %s times prediction fraud error' %fraud_count_prediction_error)
#    if fraud_count_prediction:
#        print('the fraud prediction accuracy is %s' %(fraud_count_prediction/fraud_count))



#参数设置
steps = 100
pitch_size = 56962



data_cr = pd.read_csv('../creditcard.csv')

data_cr['Hour'] =data_cr["Time"].apply(lambda x : divmod(x, 3600)[0]) 

#droplist = ['V8', 'V13', 'V15', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28','Time']

droplist = ['Time']

data_new = data_cr.drop(droplist, axis = 1)
data_new.shape # 查看数据的维度
col = ['Amount','Hour']
from sklearn.preprocessing import StandardScaler # 导入模块
sc =StandardScaler() # 初始化缩放器
data_new[col] =sc.fit_transform(data_new[col])#对数据进行标准化
data_new.head()

data_cr = data_new


#for i in range(4):
#start_index = pitch_size*i+1
#end_index = pitch_size*(i+1)
train_set = data_cr.loc[1:pitch_size*4]#just for test    ,should be loc[1:5000]

import os
generate_file_list = os.listdir('C:/Users/jxiong/Desktop/forpaper/生成欺诈数据/')

for each_generate_file in generate_file_list:
    generate_dataframe = pickle_load_generate_sample(each_generate_file)
    train_set = pd.concat([train_set, generate_dataframe], axis=0)
    del train_set['Time']
    final_D_para = train_part(steps, train_set)
    print(each_generate_file+'is training...')
    for score_threshold in np.arange(-0.02,0.02,0.01):
        print('-'*80)
        print('Threshold: %s' %score_threshold)
        test_set = data_cr.loc[pitch_size*4+1:]#[142405:284808]#just for test, should be loc[5001:10000]
        test_part(final_D_para, test_set, score_threshold)




    

    
    
    





    
