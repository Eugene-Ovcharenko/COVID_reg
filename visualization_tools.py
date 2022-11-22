import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# style settings
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set(font_scale=1)
sns.set_style(
    'white', {
        'xtick.bottom': True,
        'xtick.top': False,
        'ytick.left': True,
        'ytick.right': False,
        'font.family': 'sans serif',
        'font.sans-serif': 'Arial',
        'font.style': 'bold',
    }
)


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
