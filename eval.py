import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


test_scores = np.load('scores/test_scores.npy')
train_scores = np.load('scores/train_scores.npy')

zeros = np.zeros_like(test_scores)
ones = np.ones_like(train_scores)

y_score = np.concatenate([test_scores, train_scores])
y_true = np.concatenate([zeros, ones])

fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)

sns.lineplot(x=[0, 1], y=[0, 1], color='black', alpha=0.5, linestyle='--')
sns.lineplot(x=fpr, y=tpr, errorbar=None)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('results/roc_curve.png')
