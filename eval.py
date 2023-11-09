import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


FOLDER = 'experiments/gpt2-clm-ag-news-1000'


def plot_roc(folder, test_scores, train_scores):
    # sklearn.metrics.roc_curve uses thresholds as bottom limits, so it checks whether `score > threshold`.
    # However, from the paper, `score <= threshold` predicts training examples while `score > threshold` predicts test examples.
    # Thus, the `pos_label` must be associated with the test examples, not the training examples.
    ones = np.ones_like(test_scores)
    zeros = np.zeros_like(train_scores)

    y_score = np.concatenate([test_scores, train_scores])
    y_true = np.concatenate([ones, zeros])

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)

    sns.lineplot(x=[0, 1], y=[0, 1], color='black', alpha=0.5, linestyle='--')
    sns.lineplot(x=fpr, y=tpr, errorbar=None)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(os.path.join(folder, 'roc_curve.png'))

    data = []
    for r in (0.0001, 0.001, 0.01):
        index = len(fpr[fpr < r]) - 1
        data.append(f'{tpr[index] * 100:.02f}% TPR at {r * 100:.02f}% FPR')
    data.append(f'AUC: {roc_auc_score(y_true, y_score)}')
    metrics = '\n'.join(data)
    print(metrics)
    with open(os.path.join(folder, 'metrics.txt'), 'w') as file:
        file.write(metrics + '\n')


test_scores = np.load(os.path.join(FOLDER, 'test_scores.npy'))
train_scores = np.load(os.path.join(FOLDER, 'train_scores.npy'))
plot_roc(FOLDER, test_scores, train_scores)
