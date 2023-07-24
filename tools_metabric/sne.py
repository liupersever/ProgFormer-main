import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


X1 = torch.load('train_data/X_0.pkl')
X2 = torch.load('train_data/X_1.pkl')
X3 = torch.load('train_data/X_2.pkl')
X4 = torch.load('train_data/X_3.pkl')
X5 = torch.load('train_data/X_4.pkl')
X = torch.cat((X1, X2, X3, X4, X5), dim=0)

y1 = torch.load('train_data/fake_pred1.pkl').tolist()
# print(y1)
y2 = torch.load('train_data/fake_pred2.pkl').tolist()
# print(y2)
y3 = torch.load('train_data/fake_pred3.pkl').tolist()
# print(y3)
y4 = torch.load('train_data/fake_pred4.pkl').tolist()
# print(y4)
y5 = torch.load('train_data/fake_pred5.pkl').tolist()
# print(y5)

y = y1 + y2 + y3 + y4 + y5



for i in range(len(y)):
    if y[i] == 1:
        y[i] = 'Long time survivor'
    else:
        y[i] = 'Short time survivor'
plt.figure(figsize=(5, 5), dpi=600)
ax = plt.subplot(111)
ax.set_title("t-SNE plot for METABRIC", fontsize=14)
# 初始化 t-SNE 模型并进行降维
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)

# markers = {'Short time survivor': "X", 'Long time survivor': "s"}
ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')
sns.scatterplot(x=X_norm[:, 0], y=X_norm[:, 1], hue=y, style=y,  legend='full')
plt.savefig('sne.png', dpi=600)
plt.savefig('sne.svg', dpi=600)
plt.show()
# # 绘制降维结果
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=plt.cm.get_cmap('jet', 10))
# plt.colorbar(ticks=range(10))
# plt.title('t-SNE visualization of MNIST')
# plt.xlabel('t-SNE feature 1')
# plt.ylabel('t-SNE feature 2')
# plt.show()
