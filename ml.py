import warnings
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, average_precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from models import decision_tree_cl
from models import random_forest_cl
from models import skl_knn_cl
from models import skl_perceptron_cl
from models import skl_mlp_cl
from models import skl_ada_boost_cl
from models import skl_bagging_cl
from models import skl_gb_cl
from models import xg_boost
from models import catboost_cl
from models import lightgbm_cl




if __name__ == '__main__':
    df = pd.read_excel('dataset/prepared_data.xlsx', sheet_name='prepared_data', index_col=0)

    X = df.copy().drop(columns=['Target', 'Clinic', 'Leukocytes day 1'])
    y = df['Target'].copy()


    clf = LogisticRegression(random_state=0, max_iter=10000)

    scores = {}
    groups = df['Clinic'].unique()
    for group in groups:
        X_test = X[df['Clinic'] == group]
        X_train = X[df['Clinic'].isin(groups[groups != group])]
        y_test = y[df['Clinic'] == group]
        y_train = y[df['Clinic'].isin(groups[groups != group])]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        metric_roc_auc = roc_auc_score(y_test, y_prob[:, 1])
        metric_f1 = f1_score(y_test, y_pred)
        metric_acc = accuracy_score(y_test, y_pred)
        metric_prec = precision_score(y_test, y_pred)
        metric_rec = recall_score(y_test, y_pred)

        scores.setdefault('ROC_AUC', []).append(metric_roc_auc)
        scores.setdefault('F1', []).append(metric_f1)
        scores.setdefault('ACC', []).append(metric_acc)
        scores.setdefault('PREC', []).append(metric_prec)
        scores.setdefault('REC', []).append(metric_rec)

    scores.setdefault('ROC_AUC_mean', []).append(np.mean(scores['ROC_AUC']))
    scores.setdefault('F1_mean', []).append(np.mean(scores['F1']))
    scores.setdefault('ACC_mean', []).append(np.mean(scores['ACC']))
    scores.setdefault('PREC_mean', []).append(np.mean(scores['PREC']))
    scores.setdefault('REC_mean', []).append(np.mean(scores['REC']))

    print(scores)
    print('ROC_AUC_mean:{:.2f}'.format(scores['ROC_AUC_mean'][0]))
    print('F1_mean:{:.2f}'.format(scores['F1_mean'][0]))
    print('Accuracy_mean:{:.2f}'.format(scores['ACC_mean'][0]))
    print('Presision_mean:{:.2f}'.format(scores['PREC_mean'][0]))
    print('Recall_mean:{:.2f}'.format(scores['REC_mean'][0]))

