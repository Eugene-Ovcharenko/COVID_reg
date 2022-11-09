import os
import warnings
import pandas as pd
import numpy as np
import re
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, List, Literal

from fontTools.feaLib import location
from supervised import AutoML
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, accuracy_score, precision_score, average_precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from lightgbm import Booster
import lightgbm as lgb

# style settings
warnings.filterwarnings("ignore")
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', None)
sns.set_context("talk")  # \ "paper" \ "poster" \ "notebook"
# sns.set_style("whitegrid")
# plt.style.use('seaborn')
sns.set(font_scale=1)
sns.set_style('white', {'xtick.bottom': True, 'xtick.top': False, 'ytick.left': True, 'ytick.right': False,
                        'axes.spines.left': True, 'axes.spines.bottom': True, 'axes.spines.right': True,
                        'axes.spines.top': True, 'font.family': 'sans serif', 'font.sans-serif': 'Arial',
                        'font.style': 'bold'})
cmap = plt.cm.get_cmap('coolwarm') # YlGnBu
cmap2 = plt.cm.get_cmap('tab10')

def get_leaderboard(
        fpath: str,
        names: list[str],
        load_mode: Literal["all", "uniques"] = "uniques"
) -> pd.DataFrame:
    '''
    Read the AutoML mljar-supervised results function.
    In current version Golden Features, SelectedFeatures, Stacked and Essembly models drop !!!

    :param path: name of AutoML folder,
    :param names: list of Automl subfolder names,
    :param load_mode: 'all' (all best models), "uniques" (only one model for each class).
    :return: leaderboard models DataFrame
    '''
    leaderboard = pd.DataFrame()
    for name in names:
        path = os.path.join(fpath, name)
        df = pd.read_csv(path + '\leaderboard.csv')
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

    print('\nModels leaderboard :\n', '-' * 200, '\n',
          leaderboard[['model_type', 'features_options', 'metric_type', 'metric_value', 'name', 'path']],
          '\n', '-' * 200)
    leaderboard.to_excel('results\AutoML\leaderboard.xlsx')
    return leaderboard


def model_extractor(
        path: str
) -> list:
    '''
    Extracting fitted models from the given path for each cv_fold

    :param path: path of the model to extract,
    :return: model list for each cv_fold
    '''

    models = []
    for file in os.listdir(path):
        if file.startswith('learner_fold_') and \
                file.endswith(('.baseline', '.linear', '.k_neighbors', '.decision_tree', '.extra_trees',
                               '.random_forest', '.xgboost', '.catboost', '.lightgbm', '.neural_network')):
            f_path = path + '\\' + file

            if file.endswith('.xgboost'):
                model = XGBClassifier()
                model.load_model(f_path)
            elif file.endswith('.catboost'):
                model = CatBoostClassifier()
                model.load_model(f_path)
            elif file.endswith('.lightgbm'):
                # model = lgb.LGBMClassifier(model_file=f_path)
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

    :param path: path of the folder of CV folds to extract,
    :return: model list for each cv_fold
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

    :param path: path of model,
    :return: shap_values: SHAP importance values for each cv_fold
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
                values.columns = ['mean_importance','fold']
                values['mean_importance'] = values['mean_importance'].astype(float)
                fe_values = pd.concat([fe_values, values], axis=0)

    fe_values.reset_index(inplace=True)
    return fe_values


def data_preprocessing(
        path: str,
        X: pd.DataFrame
)-> pd.DataFrame:
    '''
    Function read the AutoML MLJAR framework and preprocess the data

    :param path: path of the MLJAR framework.json
    :return: Scaled X DataFrame
    '''
    path = path + '\\framework.json'
    with open(path, 'r') as f:
        framework = json.load(f)

    # columns scale
    col_scaled = framework['params']['preprocessing']['columns_preprocessing']
    col_scaled = [col for col in col_scaled if col_scaled[col] == ['scale_normal']]
    print('X scaled column: {} \ {}'.format(len(col_scaled), len(X.columns)), end='\t')
    if col_scaled != []:
        scaler = StandardScaler()
        X[col_scaled] = scaler.fit_transform(X[col_scaled])

    # drop features
    if 'drop_features' in framework['preprocessing'][0]:
        drop_features = framework['preprocessing'][0]['drop_features']
        X.drop(drop_features, axis=1, inplace=True)
        print('dropped features: {} \ {}'.format(len(drop_features), len(X.columns)))
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

    :param y: Ground Truth y
    :param y_pred: predicted values by X data
    :param y_prob: probability of predictions by X data
    :param cv: numpy array of CV folds: rows train_fold_0, validation_fold_0...
    :return: DataFrame of classifier evaluation metrics
    """
    metric_roc = roc_auc_score(y, y_prob, average=None)
    metric_acc = accuracy_score(y, y_pred)
    metric_prec = precision_score(y, y_pred)
    metric_prec_ave = average_precision_score(y, y_pred)
    metric_rec = recall_score(y, y_pred)
    metric_f1 = f1_score(y, y_pred)
    metric_specificity = recall_score(y, y_pred, average=None)[0]


    metrics = pd.Series({
        'ROC_AUC': metric_roc,
        'Accuracy': metric_acc,
        'Precision': metric_prec,
        'Average_Precision': metric_prec_ave,
        'Recall': metric_rec,
        'Specificity': metric_specificity,
        'F1_score': metric_f1,
    }, name = 'all_folds')
    if cv is None:
        metrics=pd.DataFrame(metrics)
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
            fold_metrics = pd.Series({
                'ROC_AUC': metric_roc,
                'Accuracy': metric_acc,
                'Precision': metric_prec,
                'Average_Precision': metric_prec_ave,
                'Recall': metric_rec,
                'Specificity': metric_specificity,
                'F1_score': metric_f1
            }, name=fold)
            metrics = pd.concat([metrics, fold_metrics], axis=1)

    return metrics.T


def custom_confusion_matrix(
        y: pd.Series,
        y_pred: pd.Series,
        path: str
) -> None:
    """
    Custom confusion matrix for the given y_Ground_Truth and y_predicted

    :param y_pred: predicted values by X data
    :param y_prob: probability of predictions by X data
    :param path: path with file_name for the figure
    """
    cf_matrix = confusion_matrix(y, y_pred)
    fig = plt.figure()
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ['{0: 0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1} \n{v2} \n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap=cmap)
    plt.tight_layout()
    fig.savefig(path, dpi=300, format='tif')


def multi_roc_curves(
        roc_ftpr: pd.DataFrame,
        path: str,
        format: Literal['dot', 'rus'] = 'dot'
) -> None:
    """
    Plot ROC-curves for all models for one target

    :param roc_ftpr: True & False Positive Rate for al models
    :param path: the directory for images save
    :param format: 'dot' - 0.0 value format, 'rus' - 0,0 value format
    """
    fig = plt.figure(figsize=(7, 6))
    if format == 'dot':
        for i, roc in roc_ftpr.iterrows():
            plt.plot(
                roc.fpr,
                roc.tpr,
                color=cmap(i / len(roc_ftpr)),
                lw=2,
                label=roc.model + " (AUC = %0.2f)" % roc.roc_score
            )
    elif format == 'rus':
        for i, roc in roc_ftpr.iterrows():
            plt.plot(
                roc.fpr,
                roc.tpr,
                color=cmap(i / len(roc_ftpr)),
                lw=2,
                label='{} (AUC={})'.format(roc.model, str(round(roc.roc_score, 3)).replace('.', ','))
            )
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)
    plt.title("ROC ", fontsize=18)
    plt.legend(loc="lower right")  # prop={'size': 11}

    if format == 'rus':
        ax = plt.gca()
        xticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], len(ax.get_xticklabels()))
        xticks = [str(round(x, 2)).replace('.', ',') for x in xticks]
        ax.set_xticklabels(xticks)
        yticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], len(ax.get_yticklabels()))
        yticks = [str(round(y, 2)).replace('.', ',') for y in yticks]
        ax.set_yticklabels(yticks)

    plt.tight_layout()
    path = 'results\AutoML\\roc_curves_' + '.tiff'
    fig.savefig(path, dpi=300, format='tif')


if __name__ == '__main__':

    # main params
    names=['Automl_Compete_auc_3clinics_max',
           'Automl_Compete_auc_3clinics_max_best',
           'Automl_Optuna_auc_3clinics_max']

    # make folders for results store
    path = 'Results\AutoML'
    os.makedirs(path, exist_ok=True)

    # load prepared data
    df = pd.read_excel('dataset/prepared_data.xlsx', sheet_name='prepared_data', index_col=0)

    # define X & y
    df.reset_index(inplace=True, drop=True)
    y_df = df['Target']
    X_df = df.drop(columns=['Target', 'Clinic', 'Leukocytes day 1'])

    # create leaderboard of models
    leaderboard = get_leaderboard(fpath='', names=names, load_mode='uniques')
    ldb_metrics = pd.DataFrame() # results of the models testing
    roc_ftpr = pd.DataFrame() # df for roc curves dumping

    # model loop
    for _, m in leaderboard.iterrows():
        print('Model:', m.model_type)

        # model extract
        path = m['path'] + '\\' + m['name']
        models = model_extractor(path)
        model_class = models[0].__class__.__name__
        predictions = pd.DataFrame()

        # data preprocessing
        X = X_df.copy()
        y = y_df.copy()
        X = data_preprocessing(path, X)

        # load stored NumPy folds for each model
        folds = cv_folds_extractor(m['path'])
        folds = folds[folds.index.str.contains('validation_fold_')].dropna(axis=1, how='all')
        # CV folds indices preparing
        cv_idx = []
        groups = df['Clinic'].unique()

        # cv folds loops from automl folds folder
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

            path = 'results\AutoML\confusion_matrix_' + m['name'] + '_' + idx + '.tiff'
            custom_confusion_matrix(y_val, y_pred, path)

        # dump the predictions and probabilities
        print('OK!\n')
        path = 'results\AutoML\predictions_' + m.model_type + '.xlsx'
        predictions.to_excel(path)

        # confusion matrix
        path = 'results\AutoML\confusion_matrix_' + m['name'] + '.tiff'
        custom_confusion_matrix(predictions.y_val, predictions.y_pred, path)

        # model evaluation
        cv = cv_folds_extractor(m['path'])
        metriсs = classifier_evaluation(predictions.y_val, predictions.y_pred, predictions.y_prob, cv)
        metriсs['Model class'] = model_class
        metriсs['Model name'] = m['name']
        metriсs['Train time'] = m['train_time']
        metriсs['AutoML_metric'] = m['metric_value']
        metriсs['path'] = m['path']
        ldb_metrics = pd.concat([ldb_metrics, metriсs], axis=0, sort=False, ignore_index=False)

        # ROC evaluation
        fpr, tpr, thresholds = roc_curve(predictions.y_val, predictions.y_prob)
        roc_ftpr = roc_ftpr.append({
            'model': model_class,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'roc_score': metriсs['ROC_AUC']
        }, ignore_index=True)

        # feature importance analysis
        path = m['path'] + '\\' + m['name']
        fi_values = feature_importance_extractor(path)

        if 'feature_importance' in globals():
            fi_values['Model'] = m['model_type']
            feature_importance = pd.concat([feature_importance, fi_values], axis=0)
        else:
            feature_importance = pd.DataFrame()

    path = 'results\AutoML\\feature_importance.xlsx'
    feature_importance.to_excel(path)
    fig = plt.figure()
    ax = sns.stripplot(
        data=feature_importance,
        x="mean_importance",
        y='index',
        hue='fold',
        s=2, dodge=True, jitter=False)
    plt.legend(title='', loc='upper right', labels=['Clinic 1', 'Clinic 2', 'Clinic 3'])
    plt.tight_layout()
    fig.savefig('results\AutoML\\feature_importance.tiff', dpi=300, format='tif')

    fig = plt.figure()
    ax = sns.barplot(
        data=feature_importance,
        x="mean_importance",
        hue='fold',
        y="index",
        linewidth=0.1, errorbar="sd", errwidth=0.5, capsize=0.2, dodge=True)
    plt.legend(title='', loc='upper right', labels=['Clinic 1', 'Clinic 2', 'Clinic 3'])
    plt.tight_layout()
    fig.savefig('results\AutoML\\feature_importance2.tiff', dpi=300, format='tif')

    fig, axn = plt.subplots(1, 3, figsize=(10,10))
    for i, ax in enumerate(axn.flat):
        print(feature_importance)
        foldid = 'learner_fold_' + str(i)
        pt = feature_importance[feature_importance['fold']==foldid].pivot_table(index='index', columns='Model', values='mean_importance')
        ax.title.set_text(foldid)
        cbar_ax = fig.add_axes([1.10, 0.5, .05, .5])
        sns.heatmap(pt, ax=ax, cmap='RdPu', annot=False, square=True, linewidths=0.5, cbar=i == 0, vmin=0, vmax=1, cbar_ax=None if i else cbar_ax)
    fig.tight_layout()
    path = 'results\AutoML\\feature_importance_matrix.tiff'
    fig.savefig(path, dpi=200, format='tif')

    # export results
    path = 'results\AutoML\\results.xlsx'
    ldb_metrics.to_excel(path, header=True)

    # chart of model's metrics by metric
    for metric in ['ROC_AUC', 'Accuracy', 'Precision', 'Average_Precision', 'Recall', 'Specificity', 'F1_score']:

        # export metrics pivot tables
        ptable = pd.pivot_table(ldb_metrics, columns=['Model class'], index=ldb_metrics.index, values=metric)
        path = 'results\AutoML\\ptable_' + metric + '.xlsx'
        ptable.to_excel(path)

        # charts of metrics
        fig = plt.figure(figsize=(15,5))
        df_reduced = ldb_metrics[ldb_metrics.index != 'all_folds']
        order_by_all_sample = \
            ldb_metrics[ldb_metrics.index == 'all_folds'].sort_values(metric, ascending=False)['Model class'].values
        order_by_mean_metric =\
            ptable[ptable.index != 'all_folds'].mean(axis=0).sort_values(ascending=False).index.tolist()
        ax = sns.barplot(
            data=df_reduced,
            x=df_reduced['Model class'],
            y=metric,
            hue=df_reduced.index,
            order=order_by_mean_metric,
            width=0.75,
            alpha=0.75,
            palette=['#2d79aa','#ff8f19','#ff5558']
        )
        plt.title(metric)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=3)
        plt.tight_layout()
        path = 'results\AutoML\\model_for_each_metric_' + metric + '.tiff'
        fig.savefig(path, dpi=200, format='tif')

    # Accuracy-Precision figure
    fig = plt.figure()
    df_reduced = ldb_metrics[ldb_metrics.index != 'all_folds']

    sns.scatterplot(
        data=df_reduced,
        x="Specificity",
        y="Recall",
        hue=df_reduced.index,
        palette=['#2d79aa', '#ff8f19', '#ff5558'],
        alpha=0.75,
        size='Accuracy',
        sizes=(20, 200)
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(bbox_to_anchor=(1.0, 1.0), fancybox=True, shadow=False)
    plt.tight_layout()
    path = 'results\AutoML\\Specifity_Recall_Accuracy.tiff'
    fig.savefig(path, dpi=200, format='tif')

    # ROC curves dump & plot
    path = 'results\AutoML\ROC_curves.xlsx'
    roc_ftpr.to_excel(path)
    path = os.path.join('AutoML', 'Analyzed', 'charts')
    multi_roc_curves(roc_ftpr, path=path, format='dot')


