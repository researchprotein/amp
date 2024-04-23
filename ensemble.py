import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, accuracy_score, auc, matthews_corrcoef, confusion_matrix

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
    Metrics_ = pd.DataFrame(Results, columns=["ACC", "MCC", "Sn", "Sp", "AUC"])

    return Metrics_

label = [1] * 1536 + [0] * 1536
label = np.array(label)

#ratio = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#ratio = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
ratio = [0.8]
# ratio = []
# val = 0
# while val <= 1.0:
#     ratio.append(val)
#     val += 0.01

# best version cnn_512_0.001_1.pkl mlp_16_0.0001_1.pkl
for rate in ratio:
    f1 = open('cnn.txt', 'r')
    f2 = open('mlp.txt', 'r')
    text1 = f1.readlines()
    text2 = f2.readlines()

    length = len(text1)
    p1 = [float(x.strip('\n')) for x in text1]
    p2 = [float(x.strip('\n')) for x in text2]
    p3 = [0] * length
    pred = [0] * length
    for i in range(length):
        p3[i] = (1 - rate) * p1[i] + rate * p2[i]
        pred[i] = 1 if p3[i] > 0.5 else 0

    print(getMetrics(label, np.array(pred), np.array(p3)), rate)
    pd.DataFrame(np.array(p3)).to_csv('ensemble.csv', header=None, index=None)