import warnings
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# style settings
warnings.filterwarnings("ignore")
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', None)
sns.set_context("talk")  # "talk" \ "paper" \ "poster" \ "notebook"
# sns.set_theme(style="ticks")
# sns.set(font_scale=1)
# sns.set_style('white', {'xtick.bottom': True, 'xtick.top': False, 'ytick.left': True, 'ytick.right': False,
#                         'axes.spines.left': True, 'axes.spines.bottom': True, 'axes.spines.right': True,
#                         'axes.spines.top': True, 'font.family': 'sans serif', 'font.sans-serif': 'Arial',
#                         'font.style': 'bold'})

colors = 'Set1'
sns.set_palette(colors)


def cor_plot(
        df: pd.DataFrame,
        withnums: bool = False
) -> None:
    '''
    Plot the correlation matrix from the DataFrame

    :param df: DataFrame of the predictors,
    :param withnums: whether to plot the correlation matrix with values
    '''

    os.makedirs('results', exist_ok=True)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), 0)
    fig, ax = plt.subplots(figsize=(16, 8))
    palette = sns.diverging_palette(240, 10, n=9, as_cmap=True)
    ax = sns.heatmap(corr, mask=mask, cmap=palette, annot=withnums,
                     vmax=1.0, vmin=-1.0, center=0, square=True,
                     linewidths=0.5, cbar_kws={"shrink": 0.5},
                     annot_kws={"size": 20 / np.sqrt(len(corr))})
    ax.set_xticks((ax.get_xticks()[:-1]))
    ax.set_yticks((ax.get_yticks()[1:]))

    plt.tight_layout()
    fig.savefig('results\CorrelationMatrix.tiff', dpi=300, format='tif')



if __name__ == '__main__':

    # load the dataset
    df = pd.read_excel('dataset/1.3. data COVID-19.xlsx', sheet_name='United')
    print('Check the duplicates:\n',
          df[df['ID'].duplicated()]['ID'])
    df.set_index('ID', inplace=True)

    # data description
    print(df.info())
    print(df.nunique())

    # data
    data = df.copy()

    # fill NA/NaN values
    df = df.fillna(df.median(), axis=0)

    # replace string classes to 1|0
    df = df.replace('yes', 1)
    df = df.replace('no', 0)
    df = df.replace('m', 1)
    df = df.replace('f', 0)
    dummies = pd.get_dummies(df['Severity'], prefix='Severity')
    df = pd.concat([df, dummies], ignore_index=False, axis=1)
    df = df.drop(columns=['Severity'])

    # # correlation matrix
    cor_plot(df)

    # export of results
    df.to_excel('dataset/prepared_data.xlsx', sheet_name='prepared_data')

    # data visualization
    intervals = data.select_dtypes(include='float64')
    categorial = data.select_dtypes(include='object')
    for cat in categorial:
        f = plt.figure(figsize=(5, 5))
        ax = sns.countplot(
            x=cat,
            data=data,
            alpha=0.7
        )
        plt.tight_layout()
        file = 'results\categorial_data_' + cat + '.tiff'
        f.savefig(file, dpi=100, format='tif')

    for i in intervals:
        f1 = plt.figure(figsize=(7, 5))
        ax1 = sns.kdeplot(
            data=data,
            x=i,
            hue="Clinic",
            fill=True,
            palette=colors,
            common_norm=False,
            alpha=0.2,
            linewidth=1,
            bw_adjust=.2
        )
        file1 = 'results\interval_data_kde_' + i + '.tiff'
        f1.savefig(file1, dpi=100, format='tif')

        f2 = plt.figure(figsize=(7, 5))
        ax2 = sns.histplot(
            data=data,
            x=i,
            hue="Clinic",
            palette=colors,
            alpha=0.2,
            element='step'
        )
        file2 = 'results\interval_data_hist_' + i + '.tiff'
        f2.savefig(file2, dpi=100, format='tif')

        for col in ['Sex', 'Target', 'Severity', 'HT', 'Diabetes', 'CAD CHF', 'COPD Asthma', 'CKD 3to5']:
            f3 = sns.displot(
                data=data,
                x=i,
                hue="Clinic",
                col=col,
                kind="hist",
                palette=colors,
                fill=True,
                alpha=0.2,
                element='step'
            )
            file3 = 'results\interval_data_dis_' + i + '_' + col + '.tiff'
            f3.savefig(file3, dpi=100, format='tif')
