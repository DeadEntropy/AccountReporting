import configparser

import pandas as pd
import glob
import os
import re
from config.config_helper import parse_list
from account_transforms import static_data as sd


def can_handle(path_in, config):
    df = pd.read_csv(path_in, nrows=1, index_col=False)
    expected_columns = parse_list(config['expected_columns'])
    return set(df.columns) == set(expected_columns)


def simplify_memo(memo):
    if memo.startswith('Transfer From Checking'):
        return 'Transfer From Checking'
    if memo.startswith('Transfer to Checking'):
        return 'Transfer to Checking'
    if memo.startswith('Transfer to MasterCard'):
        return 'Transfer to MasterCard'
    if memo.startswith('Transfer From Money Market'):
        return 'Transfer From Money Market'
    if memo.startswith('Transfer to Money Market'):
        return 'Transfer From to Market'
    if memo.startswith('Transfer From Savings Plus'):
        return 'Transfer From Savings Plus'
    if memo.startswith('Transfer to Savings Plus'):
        return 'Transfer to Savings Plus'
    if memo.startswith('ACH Electronic Debit - CHASE CREDIT CRD EPAY'):
        return 'ACH Electronic Debit - CHASE CREDIT CRD EPAY'
    if memo.startswith('Debit Card Purchase') and '#2791' in memo:
        return memo.split('#2791')[1]
    if memo.startswith('Cash Withdrawal') and '#2791' in memo:
        return 'Cash Withdrawal' + memo.split('#2791')[1]

    return memo


def load(path_in, config):
    df = pd.read_csv(path_in, sep=',', index_col=False)
    expected_columns = parse_list(config['expected_columns'])
    assert set(df.columns) == set(expected_columns), f'Was expecting [{", ".join(expected_columns)}] but file columns ' \
                                                     f'are [{", ".join(df.columns)}]. (Citi)'

    df["Debit"] = df["Debit"].fillna(0)
    df["Credit"] = df["Credit"].fillna(0)
    df_out = pd.DataFrame(columns=sd.target_columns)
    df_out.Date = pd.to_datetime(df["Date"], format='%m-%d-%Y')
    df_out.Account = 'Citi'
    df_out.Currency = config['default_currency']
    df_out.Amount = df['Credit'] - df['Debit']
    df_out.Subcategory = ''
    df_out.Memo = [simplify_memo(re.sub(' +', ' ', memo)).strip() for memo in df.Description]

    return df_out


def load_save(config):
    files = glob.glob(os.path.join(config['default_folder_in'], '*.csv'))
    print(f"found {len(files)} CSV files.")
    if len(files) == 0:
        return

    df_list = [load(f, config['default_currency']) for f in files]
    for df_temp in df_list:
        df_temp['count'] = df_temp.groupby(sd.target_columns).cumcount()
    df = pd.concat(df_list)
    df.drop_duplicates().drop(['count'], axis=1).sort_values('Date', ascending=False).to_csv(config['default_path_out'], index=False)


def load_save_default():
    config = configparser.ConfigParser()
    config.read('../config/config.ini')

    load_save(config['Citi'])
