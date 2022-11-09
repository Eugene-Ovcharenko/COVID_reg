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

    eval_metric = 'auc'
    mode = 'Compete'
    text = '3clinics_max'
    path = 'Automl_' + mode + '_' + eval_metric + '_' + text

    automl = AutoML(results_path=path,
                    mode=mode,
                    model_time_limit=60 * 60,
                    start_random_models=10,
                    hill_climbing_steps=10,
                    top_models_to_improve=10,
                    golden_features=True,
                    features_selection=True,
                    train_ensemble=True,
                    stack_models=True,
                    explain_level=2,
                    eval_metric=eval_metric,
                    ml_task='binary_classification',
                    validation_strategy={'validation_type': 'custom'}
                    )

    automl.fit(X, y, sample_weight=None, cv=cv_idx)
    automl.report()
