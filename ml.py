import os
import warnings
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from supervised import AutoML

warnings.filterwarnings("ignore")


def main(
        data_path: str,
        save_dir_prefix: str,
        target: str,
        cv_flag_col: str,
        drop_X_columns: [str],
        mode: str,
        metric: str,
        golden_features: bool,
        features_selection: bool,
        train_ensemble: bool,
        stack_models: bool,
):
    # load the dataset
    df = pd.read_excel(data_path, index_col=0)
    df.reset_index(inplace=True, drop=True)
    y = df[target].copy()
    X = df.copy().drop(columns=[target, cv_flag_col])
    X = X.drop(columns=drop_X_columns)

    # CV folds indices preparing
    cv_idx = []
    groups = df[cv_flag_col].unique()
    for i, group in enumerate(groups):
        test_id = X[df[cv_flag_col] == group].index.values
        train_id = X[df[cv_flag_col].isin(groups[groups != group])].index.values
        print(f'Fold {i}: Train data: {len(train_id)} Test data: {len(test_id)}')
        cv_idx.append((train_id, test_id))

    # preliminary test on log regression
    clf = LogisticRegression(random_state=0, penalty='l2', max_iter=10000)
    score1 = cross_val_score(clf, X, y, cv=3, n_jobs=-1, scoring='f1')
    score2 = cross_val_score(clf, X, y, cv=cv_idx, n_jobs=-1, scoring='f1')
    print('\nlog_classifier in case of standard 3folds cross validation strategy:')
    print(f'F1: {list(np.around(np.array(score1), 2))}, mean F1: {np.mean(score1):.2f}')
    print('\nlog_classifier in case of custom 3 clinic cross validation strategy:')
    print(f'F1: {list(np.around(np.array(score2), 2))}, mean F1: {np.mean(score2):.2f}\n')

    path = f'{save_dir_prefix}_{mode}_{metric}'
    automl = AutoML(results_path=path,
                    mode=mode,
                    model_time_limit=60 * 60,
                    start_random_models=10,
                    hill_climbing_steps=10,
                    top_models_to_improve=10,
                    golden_features=golden_features,
                    features_selection=features_selection,
                    train_ensemble=train_ensemble,
                    stack_models=stack_models,
                    explain_level=2,
                    eval_metric=metric,
                    ml_task='binary_classification',
                    validation_strategy={'validation_type': 'custom'}
                    )

    automl.fit(X, y, sample_weight=None, cv=cv_idx)
    automl.report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoML pipeline')
    parser.add_argument('--data_path', default='dataset/prepared_data.xlsx', type=str)
    parser.add_argument('--save_dir_prefix', default='Automl_', type=str)
    parser.add_argument('--target', default='Target', type=str, help='Target column')
    parser.add_argument('--cv_flag_col', default='Clinic', type=str, help='Clinic column')
    parser.add_argument('--drop_X_columns', default=['Leukocytes day 1'], type=list)
    parser.add_argument('--mode', default='Compete', type=str, choices=['Explain', 'Perform', 'Compete'])
    parser.add_argument('--metric', default='auc', type=str, choices=['auc', 'f1'])
    parser.add_argument('--golden_features', type=bool, default=True, choices=[True, False])
    parser.add_argument('--features_selection', type=bool, default=True, choices=[True, False])
    parser.add_argument('--train_ensemble', type=bool, default=True, choices=[True, False])
    parser.add_argument('--stack_models', type=bool, default=True, choices=[True, False])
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        save_dir_prefix=args.save_dir_prefix,
        target=args.target,
        cv_flag_col=args.cv_flag_col,
        drop_X_columns=args.drop_X_columns,
        mode=args.mode,
        metric=args.metric,
        golden_features=args.golden_features,
        features_selection=args.features_selection,
        train_ensemble=args.train_ensemble,
        stack_models=args.stack_models,
    )
