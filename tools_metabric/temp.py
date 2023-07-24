import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import interp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, matthews_corrcoef, roc_auc_score, auc
from sklearn.metrics import accuracy_score
import warnings

# from models.diff_transformer import TransformerEncoder
from models.model import ProsMSDA
from utils import *

warnings.filterwarnings("ignore")
torch.manual_seed(24)
torch.cuda.manual_seed(24)
batch_size = 16
lr = 0.00001
device = torch.device('cuda:0')


def load_array(data_arrays, batch_size, is_train=True):
    """构造⼀个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


X = pd.read_csv('../data/X_metabric_cleaned1.csv')
y = pd.read_csv('../data/y_LS_metabric_cleaned1.csv')

X1 = np.array(X)
Y1 = np.array(y)
X1 = torch.from_numpy(X1).float()
Y1 = torch.from_numpy(Y1).float()

# In[18]:

acc = 0.0
f1 = 0.0
pre = 0.0
recall = 0.0
c_index = 0.0
ks = 5

kf = StratifiedKFold(n_splits=ks, random_state=24, shuffle=True)

for index, (train, test) in enumerate(kf.split(X1, Y1)):

    pass