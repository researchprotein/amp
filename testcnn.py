import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from utility import readname, readfile, readstructure
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_curve,accuracy_score,auc,matthews_corrcoef,confusion_matrix

import warnings
warnings.filterwarnings("ignore")

def getMetrics(y_true, y_pred, y_proba):
    ACC = accuracy_score(y_true, y_pred)
    MCC = matthews_corrcoef(y_true, y_pred)
    CM = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = CM.ravel()
    Sn = tp / (tp + fn)
    Sp = tn / (tn + fp)
    FPR, TPR, thresholds_ = roc_curve(y_true, y_proba)
    AUC = auc(FPR, TPR)

    Results = np.array([ACC, MCC, Sn, Sp, AUC]).reshape(-1, 5)
    #print(Results.shape)
    Metrics_ = pd.DataFrame(Results, columns=["ACC", "MCC", "Sn", "Sp", "AUC"])

    return Metrics_

def readesm(directory, file):
    files = readname(file)
    results = []
    for sequence in files:
        name = sequence[1:]
        cur = torch.load(directory + '/' + name + '.pt')['representations'][33]
        # if cur.shape[0] < 50:
        #     cur = torch.cat((cur, torch.zeros(50 - cur.shape[0], 1280)), 0)
        results.append(cur)
    return results

sequence = readfile('dataset/independent test datasets/XUAMP/XU_AMP.fasta')
structure = readstructure(sequence, 'dataset/independent test datasets/XUAMP/XU_AMP.txt')
residue = []
for i in range(len(sequence)):
    cur = []
    for j in range(0, 500, 5):
        info = [0] * 24
        info[structure[i][j] - 1] = 1
        info[20] = structure[i][j + 1]
        info[21] = structure[i][j + 2]
        info[22] = structure[i][j + 3]
        info[23] = structure[i][j + 4]
        cur.append(info)
    residue.append(cur)

sequence = readfile('dataset/independent test datasets/XUAMP/XU_nonAMP.fasta')
structure = readstructure(sequence, 'dataset/independent test datasets/XUAMP/XU_nonAMP.txt')
for i in range(len(sequence)):
    cur = []
    for j in range(0, 500, 5):
        info = [0] * 24
        info[structure[i][j] - 1] = 1
        info[20] = structure[i][j + 1]
        info[21] = structure[i][j + 2]
        info[22] = structure[i][j + 3]
        info[23] = structure[i][j + 4]
        cur.append(info)
    residue.append(cur)

svmprot = []
embedding = pd.read_csv("dataset/independent test datasets/XUAMP/XU_AMP_SVMProt.csv", header=0)
property = embedding.iloc[:, 0:188].values.tolist()
prot = [p for p in property]

for i in range(0, len(prot)):
    svmprot.append(prot[i])

embedding = pd.read_csv("dataset/independent test datasets/XUAMP/XU_nonAMP_SVMProt.csv", header=0)
property = embedding.iloc[:, 0:188].values.tolist()
prot = [p for p in property]

for i in range(0, len(prot)):
    svmprot.append(prot[i])

esm = readesm('dataset/independent test datasets/XUAMP/XU_AMP_esm', 'dataset/independent test datasets/XUAMP/XU_AMP.fasta')
node = [item for item in esm]

array = []
for i in range(0, len(node)):
    avg = node[i].numpy().mean(axis=0)
    array.append(avg.tolist())

esm = readesm('dataset/independent test datasets/XUAMP/XU_nonAMP_esm', 'dataset/independent test datasets/XUAMP/XU_nonAMP.fasta')
node = [item for item in esm]

for i in range(0, len(node)):
    avg = node[i].numpy().mean(axis=0)
    array.append(avg.tolist())

embedding = pd.read_csv("dataset/independent test datasets/XUAMP/XU_AMP.csv", header=0)
property = embedding.iloc[:, 1:2022].values.tolist()
prot = [p for p in property]

single = []
for i in range(0, len(prot)):
    single.append(prot[i])

embedding = pd.read_csv("dataset/independent test datasets/XUAMP/XU_nonAMP.csv", header=0)
property = embedding.iloc[:, 1:2022].values.tolist()
prot = [p for p in property]

for i in range(0, len(prot)):
    single.append(prot[i])

label = [1] * 1536 + [0] * 1536
test_tag = torch.from_numpy(np.array(label)).long()

dataset = []
for i in range(len(single)):
    x = residue[i]
    x = torch.FloatTensor(x)
    x = x.unsqueeze(0)

    phy = svmprot[i] + array[i] + single[i]
    phy = torch.FloatTensor(phy)
    phy = phy.unsqueeze(0)

    y = torch.FloatTensor(np.array([1, 0]) if label[i] == 0 else np.array([0, 1]))
    y = y.unsqueeze(0)

    data = Data(x=x, phy=phy, y=y)
    dataset.append(data)

test_dataset = dataset

class Net(torch.nn.Module):
    """构造GCN模型网络"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.ZeroPad2d(padding=(0, 16, 0, 0)),
            nn.Conv2d(1, 32, kernel_size=(3, 20), stride=(1, 20)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 1)),
            nn.LeakyReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(3489, 1024),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(p=0.2),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4160, 512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, data):
        x, phy, batch = data.x, data.phy, data.batch
        x = x.unsqueeze(1)

        x = self.conv(x)
        mid = self.fc1(phy)
        y = self.fc2(torch.cat((x.view(x.shape[0], -1), mid), 1))
        return y

model = Net()
batch_size = 512
learning_rate = 0.001

def evaluate(loader):
    model.eval()
    pred = []
    label = []
    with torch.no_grad():
        for data in loader:
            pred.extend(model(data).numpy().tolist())
            label.extend(data.y.numpy().tolist())
    return pred, label

loaders = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

probability = [0] * len(label)
prediction = [0] * len(label)

model = torch.load('cnn_{}_{}_1.pkl'.format(batch_size, learning_rate))
pred, cur = evaluate(loaders)
for j in range(len(cur)):
    probability[j] = pred[j][1]
    if pred[j][1] > 0.5:
        prediction[j] = 1
print(getMetrics(label, prediction, probability))
pd.DataFrame(np.array(probability)).to_csv('cnn.csv', header=None, index=None)