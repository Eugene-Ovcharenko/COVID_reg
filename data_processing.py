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


def cor_plot(df):  # Correlation visualization function
    os.makedirs('results', exist_ok=True)

    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), 1)                               #  mask for the upper triangle
    fig, ax = plt.subplots(figsize=(16, 8))
    cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)                        # colormap
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True,
                vmax=1.0, vmin=-1.0, center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.5},
                annot_kws={"size": 20 / np.sqrt(len(corr))})
    plt.tight_layout()
    fig.savefig('results\CorrelationMatrix.tiff', dpi=300, format='tif')
    return fig


if __name__ == '__main__':

    # load the dataset
    df = pd.read_excel('dataset/1.3. data COVID-19.xlsx', sheet_name='United')
    print('Check the duplicates:\n',
          df[df['ID'].duplicated()]['ID'])
    df.set_index('ID', inplace=True)

    # data description
    print(df.info())
    print(df.nunique())

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

    # correlation matrix
    cor_plot(df)

    # export of results
    df.to_excel('dataset/prepared_data.xlsx', sheet_name='prepared_data')


