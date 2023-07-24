#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
# import utils
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as Data

import matplotlib.pyplot as plt

import pickle as pkl
import pandas as pd
import numpy as np

from scipy import stats
import seaborn as sns

from matplotlib.colors import ListedColormap

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, r2_score, median_absolute_error, mean_absolute_error, accuracy_score, \
    f1_score, precision_score, recall_score
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, cross_val_predict, \
    KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import balanced_accuracy_score
from collections import OrderedDict
from itertools import combinations

from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

from torch.autograd import Variable
from torch import autograd
import warnings

warnings.filterwarnings("ignore")

batch_size = 256
lr = 0.003  ######0.003
beta1 = 0.5
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def C_index(y_true, y_pred):
    labels = np.sort(np.unique(y_true))
    pair_set = OrderedDict()
    for label_pair in combinations(labels, 2):
        pair_set[label_pair] = []
    for idx_pair in combinations(range(len(y_true)), 2):
        if y_true[idx_pair[0]] < y_true[idx_pair[1]]:
            pair_set[(y_true[idx_pair[0]], y_true[idx_pair[1]])].append((idx_pair[0], idx_pair[1]))
        elif y_true[idx_pair[0]] > y_true[idx_pair[1]]:
            pair_set[(y_true[idx_pair[1]], y_true[idx_pair[0]])].append((idx_pair[1], idx_pair[0]))
        else:
            continue
    nPairs = 0
    nResult = 0
    for label_pair, idx_pair_list in pair_set.items():
        nPairs += len(idx_pair_list)
        for pair in idx_pair_list:
            if y_pred[pair[0]] < y_pred[pair[1]]:
                nResult += 1
            elif y_pred[pair[0]] == y_pred[pair[1]]:
                nResult += 0.5

    # print(nResult/nPairs)
    # print(accuracy_score(y_true=y_true,y_pred=y_pred))
    # print(mean_absolute_error(y_true=y_true,y_pred=y_pred))
    return nResult / nPairs






yt1 = torch.load('yt1.pkl')
y1 = torch.load('fake11.pkl')

yt2 = torch.load('yt2.pkl')
y2 = torch.load('fake12.pkl')

yt3 = torch.load('yt3.pkl')
y3 = torch.load('fake13.pkl')

yt4 = torch.load('yt4.pkl')
y4 = torch.load('fake14.pkl')

yt5 = torch.load('yt5.pkl')
y5 = torch.load('fake15.pkl')

print('c-index:', (roc_auc_score(yt1, y1) + roc_auc_score(yt2, y2) + roc_auc_score(yt3, y3) + roc_auc_score(yt4, y4) + roc_auc_score(yt5, y5)) / 5)

# print('c-index:', (roc_auc_score(yt1, np.rint(y1)) + roc_auc_score(yt2, np.rint(y2)) + roc_auc_score(yt3, np.rint(
#     y3)) + roc_auc_score(yt4, np.rint(y4)) + roc_auc_score(yt5, np.rint(y5))) / 5)
