import torch
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import numpy as np
from numpy import interp




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
torch.save(mean_tpr, 'trans_tpr.pkl')
torch.save(mean_fpr, 'trans_fpr.pkl')
torch.save(mean_auc, 'tran_auc.pkl')
ax2.plot(mean_fpr, mean_tpr, color='brown', label=r'ROC Mean (area=%0.3f)' % mean_auc, lw=1.2, alpha=.8)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.legend(loc="lower right", prop={'size': 10})
ax2.set_title("Cross Valiadation Roc curve for METABRIC", fontsize=12)
plt.savefig('roc_metabric.png', dpi=600)
plt.savefig('roc_metabric.svg', dpi=600)
# plt.show()
