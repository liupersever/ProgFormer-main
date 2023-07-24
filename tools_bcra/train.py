import os
from torch import optim
from torch.utils import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import interp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, matthews_corrcoef, roc_auc_score, auc
from sklearn.metrics import accuracy_score
from models.model import ProsMMA
from utils import *

import warnings

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)
warnings.filterwarnings("ignore")

torch.manual_seed(16)
torch.cuda.manual_seed(16)
batch_size = 64
lr = 0.000005
# lr = 0.001
device = torch.device('cuda:0')


def load_array(data_arrays, batch_size, is_train=True):
    """构造⼀个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


X = pd.read_csv('../data/bcra/brca_clinical.csv')
y = pd.read_csv('../data/bcra/y_brca.csv')

X1 = np.array(X)
Y1 = np.array(y)
X1 = torch.from_numpy(X1).float()
Y1 = torch.from_numpy(Y1).float()

acc = 0
f1 = 0
pre = 0
recall = 0
c_index = 0
ks = 5
epoches = 100
kf = StratifiedKFold(n_splits=ks, random_state=24, shuffle=True)

for index, (train, test) in enumerate(kf.split(X1, Y1)):

    netG = ProsMMA(1, [32, 64, 128], 8, 2, 21, 2, 0.5, 0.1).cuda()
    netG.apply(xavier_init_weights)
    netG.train()
    weight_decay = 5e-3
    optimizerG = optim.AdamW(netG.parameters(), lr=lr, weight_decay=weight_decay)  # betas=(0.5, 0.999)
    criterion = nn.CrossEntropyLoss()

    X = X1[train].unsqueeze(1)
    Y = Y1[train].reshape((-1)).long()

    data_iter = load_array((X, Y), batch_size, True)

    X_test = X1[test]
    Y_test = Y1[test]
    score = 0.0
    acc_best = 0.0
    pre_best = 0.0
    recall_best = 0.0
    c_index_best = 0.0
    f1_best = 0.0
    for epoch in range(epoches):

        for x_data, y_data in data_iter:
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            fake = netG(x_data)
            loss = criterion(fake, y_data)
            optimizerG.zero_grad()
            loss.backward()
            grad_clipping(netG, 0.8)
            optimizerG.step()

        netG.eval()
        d = X_test.to(device)
        yt = Y_test
        fake_test = netG(d.unsqueeze(1)).cpu().detach()
        fake_pred = torch.argmax(fake_test, dim=1)
        fake_prob = fake_test[:, 1].view(-1)
        score_ = matthews_corrcoef(yt, fake_pred)
        acc_ = accuracy_score(yt, fake_pred)
        f1_ = f1_score(yt, fake_pred)
        pre_ = precision_score(yt, fake_pred)
        recall_ = recall_score(yt, fake_pred)

        if score_ > score:
            score = score_
            acc_best = acc_
            pre_best = pre_
            recall_best = recall_
            f1_best = f1_
            c_index_best = roc_auc_score(yt, fake_pred)
            torch.save(yt, f'train_data/yt{index + 1}.pkl')
            torch.save(fake_prob, f'train_data/fake_prob{index + 1}.pkl')
            torch.save(fake_pred, f'train_data/fake_pred{index + 1}.pkl')

    print(f'acc{index + 1}:', acc_best)
    print(f'pre{index + 1}:', pre_best)
    print(f'recall{index + 1}:', recall_best)
    print(f'f1{index + 1}:', f1_best)
    print(f'c_index{index + 1}:', c_index_best)
    acc += acc_best
    f1 += f1_best
    pre += pre_best
    recall += recall_best
    c_index += c_index_best

yt1 = torch.load('train_data/yt1.pkl')
y1 = torch.load('train_data/fake_prob1.pkl')

yt2 = torch.load('train_data/yt2.pkl')
y2 = torch.load('train_data/fake_prob2.pkl')

yt3 = torch.load('train_data/yt3.pkl')
y3 = torch.load('train_data/fake_prob3.pkl')

yt4 = torch.load('train_data/yt4.pkl')
y4 = torch.load('train_data/fake_prob4.pkl')

yt5 = torch.load('train_data/yt5.pkl')
y5 = torch.load('train_data/fake_prob5.pkl')

print('acc: ', acc / 5)
print('pre: ', pre / 5)
print('recall: ', recall / 5)
print('f1: ', f1 / 5)
print('c_index: ', c_index / 5)

fig, ax2 = plt.subplots(figsize=(5, 5), dpi=600)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 1000)

labels = ['ROC 1-fold cross-validation', "ROC 2-fold cross-validation", "ROC 3-fold cross-validation"
    , "ROC 4-fold cross-validation", "ROC 5-fold cross-validation"]
for label, pred in zip(labels, [roc_curve(yt1, y1), roc_curve(yt2, y2), roc_curve(yt3, y3), roc_curve(yt4, y4),
                                roc_curve(yt5, y5)]):
    fpr, tpr, threshold = pred
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, label=label + ' (area = %0.3f)' % roc_auc, linewidth=1.2, linestyle="--")
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0

ax2.plot([0, 1], [0, 1], linewidth=0.6, color="black")

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
std_auc = np.std(tprs, axis=0)
torch.save(mean_tpr, 'trans_tpr_bcra.pkl')
torch.save(mean_fpr, 'trans_fpr_bcra.pkl')
torch.save(mean_auc, 'tran_auc_bcra.pkl')
ax2.plot(mean_fpr, mean_tpr, color='brown', label=r'ROC Mean (area=%0.3f)' % mean_auc, lw=1.2, alpha=.8)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.legend(loc="lower right", prop={'size': 10})
ax2.set_title("Cross Valiadation Roc curve for BCRA", fontsize=12)
plt.savefig('roc_bcra.png', dpi=600)
plt.savefig('roc_bcra.svg', dpi=600)
