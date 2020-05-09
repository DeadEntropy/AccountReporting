import pandas as pd
import glob
import os
import configparser
from config.config_helper import parse_list
from transforms import static_data as sd


def to_memo(row):
    if row['TRANSACTION'] == 'Interest':
        return 'Mortgage Interest'
    elif row['TRANSACTION'] == 'Bank payment':
        return 'Mortgage Repayment'
    elif row['TRANSACTION'] == 'Direct Debit':
        return 'Mortgage Repayment'
    return row['TRANSACTION']


def can_handle(path_in, config):
    df = pd.read_csv(path_in, nrows=1)
    expected_columns = parse_list(config['expected_columns'])
    return set(df.columns) == set(expected_columns)


def load(path_in, config):
    df = pd.read_csv(path_in)
    expected_columns = parse_list(config['expected_columns'])
    assert set(df.columns) == set(expected_columns), f'Was expecting [{", ".join(expected_columns)}] but file columns ' \
                                                     f'are [{", ".join(df.columns)}]. (Lloyds Mortgage)'

    df["OUT(£)"] = df["OUT(£)"].fillna(0)
    df["IN(£)"] = df["IN(£)"].fillna(0)

    df_out = pd.DataFrame(columns=sd.target_columns)

    df_out.Date = pd.to_datetime(df.DATE, format='%d/%m/%Y')
    df_out.Account = config['default_account_name']
    df_out.Currency = config['default_currency']
    df_out.Amount = df["IN(£)"] - df["OUT(£)"]
    df_out.Subcategory = df.TRANSACTION
    df_out.Memo = df.apply(lambda row: to_memo(row), axis=1)

    return df_out


def load_save(config):
    files = glob.glob(os.path.join(config['default_folder_in'], '*.csv'))
    print(f"found {len(files)} CSV files in {config['default_folder_in']}.")
    if len(files) == 0:
        return

    df_list = [load(f, config) for f in files]
    for df_temp in df_list:
        df_temp['count'] = df_temp.groupby(sd.target_columns).cumcount()
    df = pd.concat(df_list)
    df.drop_duplicates().drop(['count'], axis=1).sort_values('Date', ascending=False).to_csv(config['default_path_out'],
                                                                                             index=False)


def load_save_default():
    config = configparser.ConfigParser()
    config.read('../config/config.ini')

    load_save(config['LloydsMortgage'])
