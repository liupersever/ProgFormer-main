import torch
from models.model import ProsMSDA
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import pandas as pd

X = pd.read_csv('../data/X_metabric_cleaned1.csv')[700:1000]
y = pd.read_csv('../data/y_LS_metabric_cleaned1.csv')[700:1000]

X_test = np.array(X)
Y_test = np.array(y)

X_test = torch.from_numpy(X_test).float().cuda()
Y_test = torch.from_numpy(Y_test).float()

net = ProsMSDA(1, [32, 64, 128], 8, 1, 30, 0).cuda()
net.load_state_dict(torch.load('netG3.params'))

fake = net(X_test.unsqueeze(1)).cpu()
fake1 = torch.argmax(fake, dim=1).tolist()

np.savetxt('train_data/fake1.csv', fake1, delimiter=',', header='LS')

df = pd.read_csv('../data/Overall Survival (Months).csv')[700:1000]
y = df['Overall Survival (Months)']

Y = np.array(y)
print(accuracy_score(Y_test,fake1))
for i in range(len(Y)):
    if Y[i] > 60:
        Y[i] = 1
    else:
        Y[i] = 2
# print(len(Y))
# print('*'*50)
# print(Y)
# print('*'*50)
df['LSr'] = Y

df['LSf'] = fake1  # [410:820]

km = KaplanMeierFitter()
T = df['Overall Survival (Months)'] / 12
E = df['LSr']

LS = (df['LSf'] == 1.)

plt.figure(figsize=(5, 5), dpi=600)
ax = plt.subplot(111)

ax.set_title("KM survival curve for ProsDT", fontsize=14)

km.fit(T[LS], E[LS])
km.plot_survival_function(ax=ax, linewidth='1.0', label="Long time survivor")
km.fit(T[~LS], E[~LS])
km.plot_survival_function(ax=ax, linestyle='--', linewidth='1.0', label="Short time survivor")
ax.set_xlabel('Timeline(years)')

ax.set_ylabel('Cumulative survival (percentage)')

lr = logrank_test(T[LS], T[~LS], E[LS], E[~LS], alpha=.99)
print(lr.p_value)
ax.text(0, 0, "p-value={}".format(lr.p_value))
plt.savefig('km.png', dpi=600)
# In[7]:

