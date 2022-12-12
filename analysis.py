import os
import warnings
import re
import json
import argparse
from typing import Optional, Union, List, Literal

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, normalize
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from lightgbm import Booster
import lightgbm as lgb
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    average_precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from visualization_tools import (
    multi_roc_curves,
    custom_confusion_matrix,
    feature_importance_chart,
    feature_importance_folded_chart,
    ml_models_metric_charts,
    ml_metric_bubble_chart
)

warnings.filterwarnings("ignore")
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', None)
sns.set_context("paper")  # \ "talk" \ "poster" \ "notebook"


def get_leaderboard(
        fpath: str,
        names: list[str],
        load_mode: Literal["all", "uniques"] = "uniques"
) -> pd.DataFrame:
    '''
    Read the AutoML mljar-supervised results function.
    Golden Features, Selected Features, Stacked and Assembly models are being dropped in the current version.

    Args:
        fpath: name of AutoML folder,
        names: list of Automl subfolder names,
        load_mode: 'all' (all best models), "uniques" (only one model for each class).
    Returns:
        leaderboard models DataFrame
    '''

    leaderboard = pd.DataFrame()
    for name in names:
        path = os.path.join(fpath, name)
        path_ldb = os.path.join(path, 'leaderboard.csv')

        df = pd.read_csv(path_ldb)
        df.sort_values(by=['model_type', 'metric_value'], ascending=False, inplace=True)

        # DROP Golden Features
        df['features_options'] = df['name'].apply(
            lambda n: re.findall('GoldenFeatures', n)[0] if re.findall('GoldenFeatures', n) else "")
        df.drop(df[df['features_options'] == 'GoldenFeatures'].index, inplace=True)

        # DROP Selected Features
        df['features_options'] = df['name'].apply(
            lambda n: re.findall('SelectedFeatures', n)[0] if re.findall('SelectedFeatures', n) else "")
        df.drop(df[df['features_options'] == 'SelectedFeatures'].index, inplace=True)

        # DROP Stacked models
        df['features_options'] = df['name'].apply(
            lambda n: re.findall('Stacked', n)[0] if re.findall('Stacked', n) else "")
        df.drop(df[df['features_options'] == 'Stacked'].index, inplace=True)

        # DROP Ensemble models
        df.drop(df[df['model_type'] == 'Ensemble'].index, inplace=True)

        # DROP Duplicates and sort data
        df.drop_duplicates(subset=['model_type'], keep='first', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['path'] = path
        leaderboard = pd.concat([leaderboard, df])

        if load_mode == 'all':
            leaderboard.sort_values(by=['model_type', 'metric_value'], ascending=True, inplace=True)
            leaderboard.reset_index(drop=True, inplace=True)
        elif load_mode == 'uniques':
            leaderboard.sort_values(by=['model_type', 'metric_value'], ascending=True, inplace=True)
            leaderboard.drop_duplicates(subset=['model_type'], keep='last', inplace=True)
            leaderboard.sort_values(by=['metric_value'], ascending=True, inplace=True)
            leaderboard.reset_index(drop=True, inplace=True)
        else:
            exit('ERROR: WRONG load_mode. Must be "all" or "uniques" !')

    print('\nModels leaderboard :\n{}\n{}\n{}\n'.format(
        ('-' * 200),
        leaderboard[['model_type', 'features_options', 'metric_type', 'metric_value', 'name', 'path']],
        ('-' * 200)
    )
    )

    leaderboard.to_excel('results\\AutoML\\leaderboard.xlsx')
    return leaderboard


def model_extractor(
        path: str
) -> list:
    '''
    Extracting fitted models from the given path for each cv_fold

    Args:
        path: path of the model to extract,
    Returns:
        list OF MODELS for each cv_fold
    '''

    models = []
    for file in os.listdir(path):
        if file.startswith('learner_fold_') and \
                file.endswith((
                        '.baseline',
                        '.linear',
                        '.k_neighbors',
                        '.decision_tree',
                        '.extra_trees',
                        '.random_forest',
                        '.xgboost',
                        '.catboost',
                        '.lightgbm',
                        '.neural_network'
                )
                ):
            f_path = os.path.join(path, file)

            if file.endswith('.xgboost'):
                model = XGBClassifier()
                model.load_model(f_path)
            elif file.endswith('.catboost'):
                model = CatBoostClassifier()
                model.load_model(f_path)
            elif file.endswith('.lightgbm'):
                model = lgb.Booster(model_file=f_path)
            else:
                model = joblib.load(f_path)
            models.append(model)

    return models


def cv_folds_extractor(
        path: str
) -> pd.DataFrame:
    '''
    Extracting and return validation/train cv_folds from the path

    Args:
        path: path of the folder of CV folds to extract,
    Returns:
        arrays of  cv_fold
    '''

    path = os.path.join(path, 'folds')
    cv_folds = pd.DataFrame()

    for file in os.listdir(path):
        if file.startswith('fold_') and file.endswith('_validation_indices.npy'):
            cv_fold_num = re.search(r'fold_\d', file).group(0)
            f_path = os.path.join(path, file)
            idx = np.load(f_path)  # validation indices
            cv_folds = cv_folds.append(pd.Series(data=idx, name=('validation_' + cv_fold_num)))

        elif file.startswith('fold_') and file.endswith('_train_indices.npy'):
            cv_fold_num = re.search(r'fold_\d', file).group(0)
            f_path = os.path.join(path, file)
            idx = np.load(f_path)  # validation indices
            cv_folds = cv_folds.append(pd.Series(data=idx, name=('train_' + cv_fold_num)))

    return cv_folds


def feature_importance_extractor(
        path: str
) -> pd.DataFrame:
    '''
    Extracting and return mean SHAP importance values for each cv_fold from the path

    Args:
        path: path of model,
    Returns:
        shap_values: SHAP importance values for each cv_fold
    '''
    fe_values = pd.DataFrame()
    for file in os.listdir(path):

        if file.startswith('learner_fold_') and file.endswith('_importance.csv'):
            cv_fold_num = re.search(r'learner_fold_\d', file).group(0)
            f_path = os.path.join(path, file)
            values = pd.read_csv(f_path, index_col=0, names=[cv_fold_num])

            if (values.loc['feature'] == 'mean_importance').all():
                values.drop(['feature'], axis=0, inplace=True)
                values['fold'] = values.columns[0]
                values.columns = ['mean_importance', 'fold']
                values['mean_importance'] = values['mean_importance'].astype(float)
                fe_values = pd.concat([fe_values, values], axis=0)

    fe_values.reset_index(inplace=True)
    return fe_values


def data_preprocessing(
        path: str,
        X: pd.DataFrame
) -> pd.DataFrame:
    '''
    Function read the AutoML MLJAR framework and preprocess the data

    Args:
        path: path of the MLJAR framework.json
    Returns:
        Scaled X DataFrame
    '''
    path = os.path.join(path, 'framework.json')
    with open(path, 'r') as f:
        framework = json.load(f)

    # columns scale
    col_scaled = framework['params']['preprocessing']['columns_preprocessing']
    col_scaled = [col for col in col_scaled if col_scaled[col] == ['scale_normal']]
    print(f'X scaled column: {len(col_scaled)} \\ {len(X.columns)}', end='\t')

    if col_scaled != []:
        scaler = StandardScaler()
        X[col_scaled] = scaler.fit_transform(X[col_scaled])

    # drop features
    if 'drop_features' in framework['preprocessing'][0]:
        drop_features = framework['preprocessing'][0]['drop_features']
        X.drop(drop_features, axis=1, inplace=True)
        print(f'dropped features: {len(drop_features)} \\ {len(X.columns)}')
    else:
        print('| No dropped features found')

    return X


def classifier_evaluation(
        y: pd.Series,
        y_pred: pd.Series,
        y_prob: pd.Series,
        cv: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Evaluate the performance of the classifier by standard metrics

    Args:
        y: Ground Truth y
        y_pred: predicted values by X data
        y_prob: probability of predictions by X data
        cv: numpy array of CV folds: rows train_fold_0, validation_fold_0...
    Returns:
        DataFrame of classifier evaluation metrics
    """
    metric_roc = roc_auc_score(y, y_prob, average=None)
    metric_acc = accuracy_score(y, y_pred)
    metric_prec = precision_score(y, y_pred)
    metric_prec_ave = average_precision_score(y, y_pred)
    metric_rec = recall_score(y, y_pred)
    metric_f1 = f1_score(y, y_pred)
    metric_specificity = recall_score(y, y_pred, average=None)[0]
    metrics = pd.Series(
        {
            'ROC_AUC': metric_roc,
            'Accuracy': metric_acc,
            'Precision': metric_prec,
            'Average_Precision': metric_prec_ave,
            'Recall': metric_rec,
            'Specificity': metric_specificity,
            'F1_score': metric_f1,
        },
        name='all_folds'
    )

    if cv is None:
        metrics = pd.DataFrame(metrics)
    else:
        cv_val = cv[cv.index.str.startswith('validation_fold_')]

        for fold in cv_val.index:
            id = cv_val.loc[fold].dropna().astype(int).tolist()
            metric_roc = roc_auc_score(y[id], y_prob[id], average=None)
            metric_acc = accuracy_score(y[id], y_pred[id])
            metric_prec = precision_score(y[id], y_pred[id])
            metric_prec_ave = average_precision_score(y[id], y_pred[id])
            metric_rec = recall_score(y[id], y_pred[id])
            metric_f1 = f1_score(y[id], y_pred[id])
            metric_specificity = recall_score(y[id], y_pred[id], average=None)[0]
            fold_metrics = pd.Series(
                {
                    'ROC_AUC': metric_roc,
                    'Accuracy': metric_acc,
                    'Precision': metric_prec,
                    'Average_Precision': metric_prec_ave,
                    'Recall': metric_rec,
                    'Specificity': metric_specificity,
                    'F1_score': metric_f1
                },
                name=fold
            )
            metrics = pd.concat([metrics, fold_metrics], axis=1)

    return metrics.T


def predictions_by_folds(
        X: pd.DataFrame,
        y: pd.DataFrame,
        folds: pd.DataFrame,
        m: pd.Series,
        models: list
) -> pd.DataFrame:
    '''
    Make predictions by folds

    Args:
        X: predictors df
        y: target df
        folds: folds df
        m: model leaderboard data
        models: classifier model

    Returns:
        Predictions dataframe
    '''

    predictions = pd.DataFrame()
    print('Fold #:', end='\t')
    for idx, fold in folds.iterrows():
        i_val = fold.dropna().astype(int).values
        i_fold = int(re.search('validation_fold_(\d+)', idx).group(1))
        print(i_fold, end='-')

        X_val = X.iloc[i_val]
        y_val = y.iloc[i_val]

        if m.model_type == 'LightGBM':
            y_prob = models[i_fold].predict(X_val)
            y_pred = np.where(y_prob > 0.5, 1, 0)
            y_prob = pd.DataFrame(data=y_prob, index=y_val.index, columns=['y_prob'])
            y_pred = pd.DataFrame(data=y_pred, index=y_val.index, columns=['y_pred'])
            res = pd.concat([y_val, y_prob, y_pred], axis=1, ignore_index=False, sort=False)
            res.set_axis(['y_val', 'y_prob', 'y_pred'], axis=1, inplace=True)
            predictions = pd.concat([predictions, res], axis=0, sort=False)

        else:
            y_pred = models[i_fold].predict(X_val)
            y_prob = models[i_fold].predict_proba(X_val)[:, 1]
            y_pred = pd.DataFrame(data=y_pred, index=y_val.index, columns=['y_pred'])
            y_prob = pd.DataFrame(data=y_prob, index=y_val.index, columns=['y_prob'])
            res = pd.concat([y_val, y_prob, y_pred], axis=1, ignore_index=False, sort=False)
            res.set_axis(['y_val', 'y_prob', 'y_pred'], axis=1, inplace=True)
            predictions = pd.concat([predictions, res], axis=0, sort=False)

    print('OK!\n')
    return predictions


def main(
        data_path: str,
        target: str,
        cv_flag_col: str,
        drop_X_columns: [str],
        automl_folders: List[str],
        result_path: str,
):
    # make folders for results store
    os.makedirs(result_path, exist_ok=True)

    # load prepared data
    df = pd.read_excel(data_path, index_col=0)
    df.reset_index(inplace=True, drop=True)
    y_df = df['Target'].copy()
    X_df = df.copy().drop(columns=[target, cv_flag_col])
    X_df = X_df.drop(columns=drop_X_columns)

    # create leaderboard of models
    leaderboard = get_leaderboard(fpath='', names= automl_folders, load_mode='uniques')

    # loop for models
    for _, m in leaderboard.iterrows():
        print(f'Model:{m.model_type}')

        # model extract
        path = os.path.join(m['path'], m['name'])
        models = model_extractor(path)
        model_class = models[0].__class__.__name__

        # data preprocessing
        X = X_df.copy()
        y = y_df.copy()
        X = data_preprocessing(path, X)

        # load stored NumPy folds for each model
        folds = cv_folds_extractor(m['path'])
        folds = folds[folds.index.str.contains('validation_fold_')].dropna(axis=1, how='all')

          # cv folds loops from automl folds folder
        predictions = predictions_by_folds(X, y, folds, m, models)
        fname = 'predictions_' + m.model_type + '.xlsx'
        path = os.path.join('results', 'AutoML', fname)
        predictions.to_excel(path)

        # confusion matrix
        fname = 'confusion_matrix_' + m['name'] + '.tiff'
        path = os.path.join('results', 'AutoML', fname)
        custom_confusion_matrix(predictions.y_val, predictions.y_pred, path)

        # model evaluation
        cv = cv_folds_extractor(m['path'])
        metriсs = classifier_evaluation(predictions.y_val, predictions.y_pred, predictions.y_prob, cv)
        metriсs['Model class'] = model_class
        metriсs['Model name'] = m['name']
        metriсs['Train time'] = m['train_time']
        metriсs['AutoML_metric'] = m['metric_value']
        metriсs['path'] = m['path']
        if 'ldb_metrics' not in locals():
            ldb_metrics = pd.DataFrame(metriсs)  # results of the models testing
        else:
            ldb_metrics = pd.concat([ldb_metrics, metriсs], axis=0, sort=False, ignore_index=False)

        # ROC evaluation
        if 'roc_ftpr' not in locals():
            roc_ftpr = pd.DataFrame()
        fpr, tpr, thresholds = roc_curve(predictions.y_val, predictions.y_prob)
        roc_ftpr = roc_ftpr.append({
            'model': model_class,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'roc_score': metriсs['ROC_AUC']
        }, ignore_index=True)

        # feature importance analysis
        path = os.path.join(m['path'], m['name'])
        fi_values = feature_importance_extractor(path)
        if 'feature_importance' in locals():
            fi_values['Model'] = m['model_type']
            feature_importance = pd.concat([feature_importance, fi_values], axis=0)
        else:
            feature_importance = pd.DataFrame(fi_values)

    # path of directory for results export
    path = os.path.join('results', 'AutoML')

    # export of feature importance table data
    subpath = os.path.join(path, 'feature_importance.xlsx')
    feature_importance.to_excel(subpath)

    # export of ml analysis table data
    subpath = os.path.join(path, 'results.xlsx')
    ldb_metrics.to_excel(subpath, header=True)

    # ROC curves chart
    multi_roc_curves(roc_ftpr, save_dir=path, format='dot')

    # plot feature importance chart
    feature_importance_chart(feature_importance, path, format='strip')

    # plot feature importance by folds chart
    feature_importance_folded_chart(
        feature_importance=feature_importance,
        save_dir=path,
    )

    # chart of model's metrics by metric
    list_of_metrics = ['ROC_AUC', 'Accuracy', 'Precision', 'Average_Precision', 'Recall', 'Specificity', 'F1_score']
    ml_models_metric_charts(
        ldb_metrics=ldb_metrics,
        list_of_metrics=list_of_metrics,
        save_dir=path,
        export_pivot_tables=True,
        palette=['#2d79aa', '#ff8f19', '#ff5558'],
    )

    # Accuracy-Precision figure
    ml_metric_bubble_chart(
        ldb_metrics=ldb_metrics,
        save_dir=path,
        palette=['#2d79aa', '#ff8f19', '#ff5558'],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoML result analysis module')

    parser.add_argument('--data_path', default='dataset/prepared_data.xlsx', type=str)
    parser.add_argument('--target', default='Target', type=str, help='Target column')
    parser.add_argument('--cv_flag_col', default='Clinic', type=str, help='Clinic column')
    parser.add_argument('--drop_X_columns', default=['Leukocytes day 1'], type=list)
    parser.add_argument('--automl_folders',
                        default=[
                            'Automl_Compete_auc_3clinics_max',
                            'Automl_Compete_auc_3clinics_max_best',
                            'Automl_Optuna_auc_3clinics_max'
                        ],
                        type=list
    )
    parser.add_argument('--result_path', default='Results\\AutoML', type=str)

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        target=args.target,
        cv_flag_col=args.cv_flag_col,
        drop_X_columns=args.drop_X_columns,
        automl_folders=args.automl_folders,
        result_path=args.result_path,
    )

