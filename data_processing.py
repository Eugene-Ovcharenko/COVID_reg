import os
import warnings
import argparse

import pandas as pd
import seaborn as sns

from visualization_tools import cor_plot, categorical_data_vis, interval_data_vis

# style settings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', None)
colors = 'Set1'
sns.set_palette(colors)


def data_processing(
        df: pd.DataFrame,
) -> pd.DataFrame:
    # check ID
    check_ID = df[df['ID'].duplicated()]['ID']
    print(f'Check data duplicates:\n{check_ID}\n')
    df.set_index('ID', inplace=True)

    # data description
    print('Data info:')
    print(df.info())
    print(f'Non-uniques data:\n{df.nunique()}')

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

    return df


def main(
        data_path: str,
        result_path: str,
        correlation_check: bool,
        categorical_data: bool,
        interval_data: bool
):
    # check\create result path directory
    os.makedirs(result_path, exist_ok=True)

    # load the dataset
    df = pd.read_excel(data_path)
    data = df.copy()

    # data processing
    df = data_processing(df)
    path = os.path.dirname(data_path)
    path = os.path.join(path, 'prepared_data.xlsx')
    df.to_excel(path, sheet_name='prepared_data')

    # correlation matrix
    if correlation_check is True:
        cor_plot(data=df, save_dir=result_path, withnums=True, diagonalline=False, lasttick=False)

    # categorical data visualization
    if categorical_data is True:
        categorical_data_vis(data=data, save_dir=result_path)

    # interval data visualization
    if interval_data is True:
        interval_data_vis(data=data, save_dir=result_path, kde=True, hist=True, dis=True, colorset=colors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset conversion')
    parser.add_argument('--data_path', default='dataset/data.xlsx', type=str)
    parser.add_argument('--results_path', default='results', type=str)
    parser.add_argument('--correlation_check', default=False, type=bool)
    parser.add_argument('--categorical_data', default=False, type=bool)
    parser.add_argument('--interval_data', default=False, type=bool)
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        result_path=args.results_path,
        correlation_check=args.correlation_check,
        categorical_data=args.categorical_data,
        interval_data=args.interval_data,
    )
