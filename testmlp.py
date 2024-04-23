import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sklearn.metrics import roc_curve,accuracy_score,auc,matthews_corrcoef,confusion_matrix
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

label = [1] * 1536 + [0] * 1536
test1 = []
file = open('test1.txt', 'r')
for line in file:
    line = line.strip('\n')
    test1.append(float(line))
print(getMetrics(np.array(label), np.array([1 if val > 0.5 else 0 for val in test1]), np.array(test1)))

test2 = []
file = open('test2.txt', 'r')
for line in file:
    line = line.strip('\n')
    test2.append(float(line))
print(getMetrics(np.array(label), np.array([1 if val > 0.5 else 0 for val in test2]), np.array(test2)))

test3 = []
file = open('test3.txt', 'r')
for line in file:
    line = line.strip('\n')
    test3.append(float(line))
print(getMetrics(np.array(label), np.array([1 if val > 0.5 else 0 for val in test3]), np.array(test3)))

test4 = []
file = open('test4.txt', 'r')
for line in file:
    line = line.strip('\n')
    test4.append(float(line))
print(getMetrics(np.array(label), np.array([1 if val > 0.5 else 0 for val in test4]), np.array(test4)))

test_features = []
for i in range(len(test1)):
    test_features.append([test1[i], test2[i], test3[i], test4[i]])

test_features = torch.tensor(test_features, dtype=torch.float)
label = [1] * 1536 + [0] * 1536
test_tag = torch.from_numpy(np.array(label)).long()

batch_size = 16
learning_rate = 0.0001
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, data):
        output = self.fc(data)
        return output

model = torch.load('mlp_{}_{}_1.pkl'.format(batch_size, learning_rate))
model.eval()
fit = model(test_features)
y_pred = []
y_pred_proba = []
for i in range(fit.shape[0]):
    y_pred.append(0 if fit[i][0].item() > fit[i][1].item() else 1)
    y_pred_proba.append(fit[i][1].item())
print(getMetrics(label, y_pred, y_pred_proba))
pd.DataFrame(np.array(y_pred_proba)).to_csv('mlp.csv', header=None, index=None)