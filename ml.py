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
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from supervised import AutoML

warnings.filterwarnings("ignore")


if __name__ == '__main__':

    # load the dataset
    df = pd.read_excel('dataset/prepared_data.xlsx', sheet_name='prepared_data', index_col=0)
    df.reset_index(inplace=True, drop=True)
    X = df.copy().drop(columns=['Target', 'Clinic', 'Leukocytes day 1'])
    y = df['Target'].copy()

    # CV folds indices preparing
    cv_idx = []
    groups = df['Clinic'].unique()
    for group in groups:
        test_id = X[df['Clinic'] == group].index.values
        train_id = X[df['Clinic'].isin(groups[groups != group])].index.values
        print('Train data: {}\tTest data: {}'.format(len(train_id), len(test_id)))

        cv_idx.append((train_id, test_id))

    # preliminary test on log regression
    clf = LogisticRegression(random_state=0, penalty='l2', max_iter=10000)
    score1 = cross_val_score(clf, X, y, cv=3, n_jobs=-1, scoring='f1')
    score2 = cross_val_score(clf, X, y, cv=cv_idx, n_jobs=-1, scoring='f1')
    print('F1 for log_regression cv=3: {:.2f} \t {}'.format(np.mean(score1), score1))
    print('F1 for log_regression 3*CV: {:.2f} \t {}'.format(np.mean(score2), score2))

    eval_metric = 'f1'
    mode = 'Compete'
    text = '3clinics_base'
    path = 'Automl_' + mode + '_' + eval_metric + '_' + text

    automl = AutoML(results_path=path,
                    mode=mode,
                    golden_features=False,
                    features_selection=False,
                    train_ensemble=False,
                    stack_models=False,
                    explain_level=2,
                    eval_metric=eval_metric,
                    ml_task='binary_classification',
                    validation_strategy={'validation_type': 'custom'}
                    )

    automl.fit(X, y, sample_weight=None, cv=cv_idx)






    # clf = LogisticRegression(random_state=0, max_iter=10000)

    #     y_pred = clf.predict(X_test)
    #     y_prob = clf.predict_proba(X_test)
    #
    #     metric_roc_auc = roc_auc_score(y_test, y_prob[:, 1])
    #     metric_f1 = f1_score(y_test, y_pred)
    #     metric_acc = accuracy_score(y_test, y_pred)
    #     metric_prec = precision_score(y_test, y_pred)
    #     metric_rec = recall_score(y_test, y_pred)
    #
    #     scores.setdefault('ROC_AUC', []).append(metric_roc_auc)
    #     scores.setdefault('F1', []).append(metric_f1)
    #     scores.setdefault('ACC', []).append(metric_acc)
    #     scores.setdefault('PREC', []).append(metric_prec)
    #     scores.setdefault('REC', []).append(metric_rec)
    #
    # scores.setdefault('ROC_AUC_mean', []).append(np.mean(scores['ROC_AUC']))
    # scores.setdefault('F1_mean', []).append(np.mean(scores['F1']))
    # scores.setdefault('ACC_mean', []).append(np.mean(scores['ACC']))
    # scores.setdefault('PREC_mean', []).append(np.mean(scores['PREC']))
    # scores.setdefault('REC_mean', []).append(np.mean(scores['REC']))
    #
    # print(scores)
    # print('ROC_AUC_mean:{:.2f}'.format(scores['ROC_AUC_mean'][0]))
    # print('F1_mean:{:.2f}'.format(scores['F1_mean'][0]))
    # print('Accuracy_mean:{:.2f}'.format(scores['ACC_mean'][0]))
    # print('Presision_mean:{:.2f}'.format(scores['PREC_mean'][0]))
    # print('Recall_mean:{:.2f}'.format(scores['REC_mean'][0]))

