# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 22:09:40 2018

@author: jxiong
"""

# Imports

# Numpy,Pandas
import numpy as np
import pandas as pd
import datetime

# matplotlib,seaborn,pyecharts

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# plt.style.use('ggplot')  #风格设置近似R这种的ggplot库
import seaborn as sns
sns.set_style('whitegrid')
#%matplotlib inline

# import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

#  忽略弹出的warnings
import warnings
warnings.filterwarnings('ignore')  

pd.set_option('display.float_format', lambda x: '%.4f' % x)

from imblearn.over_sampling import SMOTE
import itertools

import generate_sample_base_on_WGAN as gs

##########################################################
if_retrain = False
if_smote = True #if not generate
if_restore = False
credit_card_path = r'D:\Database\creditcard.csv'
generate_path = r'C:\Users\jxnjupt\Desktop\20180321'
model_save_path = r'C:\Users\jxnjupt\Desktop\forpaper'
generate_sample_name = 'step=99999'
train_steps = 2000
batch_size = 492
if_droplist = False
##########################################################

if if_retrain:
    gs.generate_samples(train_steps, batch_size, credit_card_path, generate_path,\
                     model_save_path, if_restore)



def pickle_load_generate_sample(generate_file):
    with open(generate_path+'/%s' %generate_file, 'rb') as f:
        fraud_sample_generated = pickle.load(f)
        fraud_sample_generated = pd.DataFrame(fraud_sample_generated, columns = \
                                              ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',\
                                               'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',\
                                               'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Hour'])
        fraud_sample_generated['Class'] = 1
        return fraud_sample_generated

data_cr = pd.read_csv(r'D:\Database\creditcard.csv')
data_cr['Hour'] =data_cr["Time"].apply(lambda x : divmod(x, 3600)[0]) #单位转换

if if_droplist:
    droplist = ['V8', 'V13', 'V15', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Time']
else:
    droplist = ['Time']
data_new = data_cr.drop(droplist, axis = 1)
#data_new.shape # 查看数据的维度

col = ['Amount','Hour']
from sklearn.preprocessing import StandardScaler # 导入模块
sc =StandardScaler() # 初始化缩放器
data_new[col] =sc.fit_transform(data_new[col])#对数据进行标准化
data_new.head()


x_feature = list(data_new.columns)
x_feature.remove('Class')
x_val = data_new[x_feature]
y_val = data_new['Class']

data_for_train = data_new.loc[:227848]
X = data_for_train[x_feature]
y = data_for_train["Class"]

n_sample = y.shape[0]
n_pos_sample = y[y == 0].shape[0]
n_neg_sample = y[y == 1].shape[0]
print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample,
                                                   n_pos_sample / n_sample,
                                                   n_neg_sample / n_sample))
print('特征维数：', X.shape[1])


if not if_smote:
    droplist.remove('Time')
    generate_dataframe = pickle_load_generate_sample(generate_sample_name).drop(droplist, axis=1)
    generate_dataframe_X = generate_dataframe.drop('Class', axis=1)
    X = pd.concat([X, generate_dataframe_X], axis=0)
    X = pd.DataFrame(X,columns=x_feature)
    y = pd.concat([y, generate_dataframe['Class']], axis=0)
    y = pd.DataFrame(y, columns=['Class'])
    print('通过WGAN方法平衡正负样本后')
    n_sample = y.shape[0]
    n_pos_sample = y[y['Class'] == 0].shape[0]
    n_neg_sample = y[y['Class'] == 1].shape[0]
    print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample,
                                                       n_pos_sample / n_sample,n_neg_sample / n_sample))
else: 
    from imblearn.over_sampling import SMOTE # 导入SMOTE算法模块
    # 处理不平衡数据
    sm = SMOTE(random_state=42)    # 处理过采样的方法
    X, y = sm.fit_sample(X, y)
    print('通过SMOTE方法平衡正负样本后')
    n_sample = y.shape[0]
    n_pos_sample = y[y == 0].shape[0]
    n_neg_sample = y[y == 1].shape[0]
    y = pd.DataFrame(y, columns=['Class'])
    print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample,
                                                       n_pos_sample / n_sample,n_neg_sample / n_sample))


clf1 = LogisticRegression() # 构建逻辑回归分类器
clf1.fit(X, y)

predicted1 = clf1.predict(X) # 通过分类器产生预测结果
print("Test set accuracy score: {:.5f}".format(accuracy_score(predicted1, y,)))

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
##################################################################################

# Compute confusion matrix
cnf_matrix = confusion_matrix(y, predicted1)  # 生成混淆矩阵
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

#Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()


y_pred1_prob = clf1.predict_proba(X)[:, 1]  # 阈值默认值为0.5

fpr, tpr, thresholds = roc_curve(y,y_pred1_prob)
roc_auc = auc(fpr,tpr)

# 绘制 ROC曲线
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.5f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # random_state = 0 每次切分的数据都一样
# 构建参数组合
X_train = X
y_train = y
#droplist.append('Class')
X_test = data_new.loc[227849:].drop('Class', axis=1)
y_test = data_new.loc[227849:]['Class']

param_grid = {'C': [0.01,0.1, 1, 10, 100, 1000,],
                            'penalty': [ 'l1', 'l2']}

grid_search = GridSearchCV(LogisticRegression(),  param_grid, cv=10) # 确定模型LogisticRegression，和参数组合param_grid ，cv指定10折
grid_search.fit(X_train, y_train) # 使用训练集学习算法


results = pd.DataFrame(grid_search.cv_results_) 
best = np.argmax(results.mean_test_score.values)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.5f}".format(grid_search.best_score_))

y_pred = grid_search.predict(X_test)
print("Test set accuracy score: {:.5f}".format(accuracy_score(y_test, y_pred,)))

print(classification_report(y_test, y_pred))

print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.5f}".format(grid_search.best_score_))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)  # 生成混淆矩阵
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()

y_pred_proba = grid_search.predict_proba(X_test)  #predict_prob 获得一个概率值

if eval(generate_sample_name[5:])<=80000 and not somte:
    thresholds = [0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]
else:
    thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]  # 设定不同阈值

plt.figure(figsize=(15,10))

j = 1
for i in thresholds:
    y_test_predictions_high_recall = y_pred_proba[:,1] > i#预测出来的概率值是否大于阈值 
    
    plt.subplot(3,3,j)
    j += 1
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_test_predictions_high_recall)
    np.set_printoptions(precision=2)

    print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

    # Plot non-normalized confusion matrix
    class_names = [0,1]
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names)
