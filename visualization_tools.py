import os
from typing import Optional, Union, List, Literal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.cm import ScalarMappable


def cor_plot(
        data: pd.DataFrame,
        save_dir: str = '',
        withnums: bool = False,
        diagonalline: bool = False,
        palette: str = 'default',
        lasttick: bool = False,
        figsize: tuple = (16, 8),
) -> None:
    ''' Plot the correlation matrix from the DataFrame

    Args:
        data: dataset,
        save_dir: directory to save Correlation_Matrix.tiff,
        withnums: whether to plot the correlation matrix with values,
        diagonalline: the diagonal line with  1,0 correlation coefficients,
        palette: 'default' to blue-red palette or matplotolib palette name,
        lasttick: turn on last tick on xy axis,
        figsize: figure size (x, y)

    Returns:
        None
    '''
    os.makedirs('results', exist_ok=True)
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), diagonalline * 1)
    if palette == 'default':
        palette = sns.diverging_palette(240, 10, n=9, as_cmap=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(
        corr,
        mask=mask,
        cmap=palette,
        annot=withnums,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot_kws={"size": 20 / np.sqrt(len(corr))})
    if lasttick is False:
        ax.set_xticks((ax.get_xticks()[:-1]))
        ax.set_yticks((ax.get_yticks()[1:]))
    plt.tight_layout()
    path = os.path.join(save_dir, 'Correlation_Matrix.tiff')
    fig.savefig(path, dpi=300, format='tif')
    plt.close(fig)


def categorical_data_vis(
        data: pd.DataFrame,
        save_dir: str = '',
        figsize: tuple = (5, 5),
) -> None:
    ''' Plot the categorical data countplot

    Args:
        data: dataset,
        save_dir: directory to save results,
        figsize: figure size (x, y)

    Returns:
        None
    '''

    categorical = data.select_dtypes(include='object')
    for cat in categorical:
        f = plt.figure(figsize=figsize)
        sns.countplot(
            x=cat,
            data=data,
            alpha=0.7
        )
        plt.tight_layout()
        file = f'categorical_data_{cat}.tiff'
        print('Make', file)
        path = os.path.join(save_dir, file)
        f.savefig(path, dpi=100, format='tif')
        plt.close(f)


def interval_data_vis(
        data: pd.DataFrame,
        save_dir: str = '',
        kde: bool = False,
        hist: bool = False,
        dis: bool = False,
        colorset: str = 'Set2',
) -> None:
    ''' Plot the interval data kdeplot|histplot|displot

    Args:
        data: dataset,
        save_dir: directory to save results,
        kde: kdeplot True/False,
        hist: histplot True/False,
        dis: displot True/False,
        colorset: matplotlib colorset name

    Returns:
        None
    '''

    intervals = data.select_dtypes(include='float64')
    for i in intervals:

        # kde plot
        if kde is True:
            f = plt.figure()
            ax = sns.kdeplot(
                data=data,
                x=i,
                hue="Clinic",
                fill=True,
                palette=colorset,
                common_norm=False,
                alpha=0.2,
                linewidth=1,
                bw_adjust=.2
            )
            file = f'interval_data_kde_{i}.tiff'
            print('Make', file)
            path = os.path.join(save_dir, file)
            f.savefig(path, dpi=100, format='tif')
            plt.close(f)

        # hist plot
        if hist is True:
            f = plt.figure()
            ax = sns.histplot(
                data=data,
                x=i,
                hue="Clinic",
                palette=colorset,
                alpha=0.2,
                element='step'
            )
            file = f'interval_data_hist_{i}.tiff'
            print('Make', file)
            path = os.path.join(save_dir, file)
            f.savefig(path, dpi=100, format='tif')
            plt.close(f)

        # displot
        if dis is True:
            interval_id = (data.nunique() <= 3) & (data.nunique() > 1)
            interval_id = interval_id.index[interval_id].tolist()
            for col in interval_id:
                f = sns.displot(
                    data=data,
                    x=i,
                    hue="Clinic",
                    col=col,
                    kind="hist",
                    palette=colorset,
                    fill=True,
                    alpha=0.2,
                    element='step'
                )
                file = f'interval_data_displot_{i}_by_{col}.tiff'
                print('Make', file)
                path = os.path.join(save_dir, file)
                f.savefig(path, dpi=100, format='tif')


def multi_roc_curves(
        roc_ftpr: pd.DataFrame,
        save_dir: str,
        palette: str = 'coolwarm',
        format: Literal['dot', 'rus'] = 'dot',
        figsize: tuple = (7, 6),
) -> None:
    ''' Plot ROC-curves for all models for one target based on roc_ftpr DataFrame

    Args:
        roc_ftpr: True & False Positive Rate for al models,
        save_dir: the directory for images save,
        palette: 'default' to blue-red palette or matplotolib palette name,
        format: 'dot' - 0.0 value format, 'rus' - 0,0 value format,
        figsize: figure size (x, y)

    Returns:
        None
    '''
    cmap = plt.cm.get_cmap(palette)

    fig = plt.figure(figsize=figsize)
    if format == 'dot':
        for i, roc in roc_ftpr.iterrows():
            plt.plot(
                roc.fpr,
                roc.tpr,
                color=cmap(i / len(roc_ftpr)),
                lw=2,
                label=f'{roc.model} (AUC = {roc.roc_score["all_folds"]:.3f})'
            )
    elif format == 'rus':
        for i, roc in roc_ftpr.iterrows():
            plt.plot(
                roc.fpr,
                roc.tpr,
                color=cmap(i / len(roc_ftpr)),
                lw=2,
                label=f'{roc.model} (AUC = {roc.roc_score["all_folds"]:.3f})'.replace('.', ',')
            )
    else:
        os.error(code='Wrong params')

    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)
    plt.title("ROC ", fontsize=18)
    plt.legend(loc="lower right")

    if format == 'rus':
        ax = plt.gca()
        xticks = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], len(ax.get_xticklabels()))
        xticks = [str(round(x, 2)).replace('.', ',') for x in xticks]
        ax.set_xticklabels(xticks)
        yticks = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], len(ax.get_yticklabels()))
        yticks = [str(round(y, 2)).replace('.', ',') for y in yticks]
        ax.set_yticklabels(yticks)

    plt.tight_layout()
    path = os.path.join(save_dir, 'roc_curves.tiff')
    fig.savefig(path, dpi=300, format='tif')


def custom_confusion_matrix(
        y: pd.Series,
        y_pred: pd.Series,
        path: str,
        cmap_name: str = 'coolwarm'
) -> None:
    """
    Custom confusion matrix for the given y_Ground_Truth and y_predicted

    Args:
        y_pred: predicted values by X data
        y_prob: probability of predictions by X data
        path: path with file_name for the figure
    Returns:
        None
    """

    cmap = plt.cm.get_cmap(cmap_name)

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


def feature_importance_chart(
        feature_importance: pd.DataFrame,
        save_dir: str,
        palette: str = 'tab10',
        format: Literal['strip', 'bars'] = 'strip',
        figsize: tuple = (6, 5),
) -> None:
    '''
    Args:
        feature_importance: feature_importance data
        save_dir: the directory for images save,
        palette: 'default' to blue-red palette or matplotolib palette name,
        format: 'strip' or 'bars' format,
        figsize: figure size (x, y) - 6x5 default
    Returns:
        None
    '''

    sns.set_palette(palette)

    fig = plt.figure(figsize=figsize)

    if format == 'strip':
        ax = sns.stripplot(
            data=feature_importance,
            x="mean_importance",
            y='index',
            hue='fold',
            s=2, dodge=True, jitter=False)
        ax.legend(title='', loc='upper right')

    elif format == 'bars':
        ax = sns.barplot(
            data=feature_importance,
            x="mean_importance",
            hue='fold',
            y="index",
            linewidth=0.1, errorbar="sd", errwidth=0.5, capsize=0.2, dodge=True)
        ax.legend(title='', loc='upper right')

    else:
        os.error(code='Wrong params')

    plt.ylabel('')
    plt.tight_layout()
    save_dir = os.path.join(save_dir, 'feature_importance.tiff')
    fig.savefig(save_dir, dpi=300, format='tif')


def feature_importance_folded_chart(
        feature_importance: pd.DataFrame,
        save_dir: str,
        palette: str = 'RdPu',
        figsize: tuple = (10, 10)
) -> None:
    '''

    Args:
        feature_importance: feature_importance data
        save_dir: the directory for images save,
        palette: 'default' to blue-red palette or matplotolib palette name,
        figsize: figure size (x, y) - 10x10 default

    Returns:
        None
    '''

    fold_number = len(feature_importance['fold'].unique())

    # heatmaps
    fig, axn = plt.subplots(1, fold_number, figsize=figsize, constrained_layout=True)
    for i, ax in enumerate(axn.flat):
        foldid = 'learner_fold_' + str(i)
        fe_fold = feature_importance[feature_importance['fold'] == foldid].pivot_table(
            index='index',
            columns='Model',
            values='mean_importance'
        )
        ax.title.set_text(foldid)
        sns.heatmap(
            fe_fold,
            ax=ax,
            cmap=palette,
            annot=False,
            square=True,
            linewidths=0.5,
            cbar=False,
            vmin=0,
            vmax=1,
            yticklabels = (i==0),
        )
        ax.set_ylabel('')
        ax.set_xlabel('')

    # color bar
    cmap = plt.get_cmap(palette)
    norm = plt.Normalize(0, 1)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axn, shrink=0.5)
    # plt.tight_layout()

    save_dir = os.path.join(save_dir, 'feature_importance_matrix.jpg')
    fig.savefig(save_dir, dpi=100, format='jpeg')


def ml_models_metric_charts(
        ldb_metrics: pd.DataFrame,
        list_of_metrics: List,
        save_dir: str,
        export_pivot_tables: bool = False,
        palette: Union[str, List] = 'tab10',
        figsize: tuple = (15, 5)
) -> None:
    '''

    Args:
        ldb_metrics: dataframe of leadearbord models metrics
        list_of_metrics: list of metrics for chart
        save_dir: the directory for images save,
        export_pivot_tables: activates the saving of results
        palette: 'default' to blue-red palette or matplotolib palette name,
        figsize: figure size (x, y) - 10x8 default

    Returns:
        None
    '''

    for metric in list_of_metrics:

        ptable = pd.pivot_table(ldb_metrics, columns=['Model class'], index=ldb_metrics.index, values=metric)

        if export_pivot_tables is True:
            fname = 'ptable_' + metric + '.xlsx'
            path = os.path.join(save_dir, fname)
            ptable.to_excel(path)

        fig = plt.figure(figsize=figsize)
        df_reduced = ldb_metrics[ldb_metrics.index != 'all_folds']

        order_by_all_sample = ldb_metrics[ldb_metrics.index == 'all_folds'].sort_values(
            metric,
            ascending=False
        )['Model class'].values

        order_by_mean_metric = ptable[ptable.index != 'all_folds'].mean(axis=0).sort_values(
            ascending=False
        ).index.tolist()

        sns.barplot(
            data=df_reduced,
            x=df_reduced['Model class'],
            y=metric,
            hue=df_reduced.index,
            order=order_by_mean_metric,
            width=0.75,
            alpha=0.75,
            palette=palette
        )
        plt.title(metric)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=3)
        plt.tight_layout()

        fname = ' ml_models_' + metric + '_chart.tiff'
        path = os.path.join(save_dir, fname)
        fig.savefig(path, dpi=200, format='tif')


def ml_metric_bubble_chart(
        ldb_metrics: pd.DataFrame,
        save_dir: str,
        palette: Union[str, List] = 'tab10',
        figsize: tuple = (5, 5)
) -> None:
    fig = plt.figure(figsize=figsize)
    df_reduced = ldb_metrics[ldb_metrics.index != 'all_folds']

    sns.scatterplot(
        data=df_reduced,
        x="Specificity",
        y="Recall",
        hue=df_reduced.index,
        palette=palette,
        alpha=0.75,
        size='Accuracy',
        sizes=(20, 200)
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc='lower left', fancybox=True, shadow=False)
    plt.tight_layout()

    save_dir = os.path.join(save_dir, 'Specifity_Recall_Accuracy.tiff')
    fig.savefig(save_dir, dpi=200, format='tif')
