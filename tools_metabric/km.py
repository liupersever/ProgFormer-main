import torch
from models.model import ProsMSDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import pandas as pd


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
net = ProsMSDA(1, [32, 64, 128], 8, 1, 30, 0).cuda()
net.load_state_dict(torch.load('netG3.params'))



kf = StratifiedKFold(n_splits=ks, random_state=24, shuffle=True)

for index, (train, test) in enumerate(kf.split(X1, Y1)):
    if index == 2:
        X_test = X1[test].cuda()
        Y_test = Y1[test]

        fake = net(X_test.unsqueeze(1)).cpu()
        fake1 = torch.argmax(fake, dim=1).tolist()

        np.savetxt('train_data/fake1.csv', fake1, delimiter=',', header='LS')




        print(accuracy_score(Y_test,fake1))

        # print(len(Y))
        # print('*'*50)
        # print(Y)
        # print('*'*50)
        df = pd.read_csv('../data/Overall Survival (Months).csv')
        y = df['Overall Survival (Months)']

        Y = np.array(y)[test]
        print(accuracy_score(Y_test, fake1))
        T = Y / 12
        for i in range(len(Y)):
            if Y[i] > 60:
                Y[i] = 1
            else:
                Y[i] = 2


        km = KaplanMeierFitter()

        E = Y

        LS = (Y == 1.)

        plt.figure(figsize=(5, 5), dpi=300)
        ax = plt.subplot(111)

        ax.set_title("KM survival curve for PregGAN", fontsize=14)

        km.fit(T[LS], E[LS])
        km.plot_survival_function(ax=ax, linewidth='1.0', label="Long time survivor")
        km.fit(T[~LS], E[~LS])
        km.plot_survival_function(ax=ax, linestyle='--', linewidth='1.0', label="Short time survivor")
        ax.set_xlabel('Timeline(years)')

        ax.set_ylabel('Cumulative survival (percentage)')

        lr = logrank_test(T[LS], T[~LS], E[LS], E[~LS], alpha=.99)
        print(lr.p_value)
        ax.text(0, 0, "p-value={}".format(lr.p_value))
        plt.savefig('km.png')
        # In[7]:

