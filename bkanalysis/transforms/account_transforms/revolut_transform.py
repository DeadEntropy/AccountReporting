import configparser

import pandas as pd
import glob
import os
from bkanalysis.config.config_helper import parse_list
from bkanalysis.transforms.account_transforms import static_data as sd


def can_handle(path_in, config):
    df = pd.read_csv(path_in, sep=';', nrows=1)
    expected_columns = parse_list(config['expected_columns'], False)

    if len(df.columns) < 2:
        return False

    currency = ''
    for ccy in parse_list(config['possible_currencies']):
        if ccy in df.columns[2]:
            currency = ccy

    return set(df.columns) == set(set([e.replace('CCY', currency) for e in expected_columns]))


def load(path_in, config):
    df = pd.read_csv(path_in, sep=';')
    expected_columns = parse_list(config['expected_columns'], False)

    for ccy in parse_list(config['possible_currencies']):
        if ccy in df.columns[2]:
            currency = ccy

    assert set(df.columns) == set(set([e.replace('CCY', currency) for e in expected_columns])), \
        f'Was expecting [{", ".join(expected_columns)}] but file columns are [{", ".join(df.columns)}]. (Nutmeg)'

    df[f" Paid In ({currency}) "] = pd.to_numeric(df[f" Paid In ({currency}) "].str.replace(',', ''),
                                                  errors='coerce').fillna(0)
    df[f" Paid Out ({currency}) "] = pd.to_numeric(df[f" Paid Out ({currency}) "].str.replace(',', ''),
                                                   errors='coerce').fillna(0)

    df_out = pd.DataFrame(columns=sd.target_columns)
    df_out.Date = pd.to_datetime(df["Completed Date "].str.strip(), format='%d %b %Y')
    df_out.Account = config['account_name'] + " " + currency
    df_out.Currency = currency
    df_out.Amount = df[f" Paid In ({currency}) "] - df[f" Paid Out ({currency}) "]
    df_out.Subcategory = df[" Category"]
    df_out.Memo = df[" Description "] + df[" Notes"] + df[" Exchange Out"] + df[" Exchange In"]
    df_out.Memo = df_out.Memo.str.replace('£', 'GBP')
    df_out.Memo = df_out.Memo.str.replace('€', 'EUR')
    df_out.Memo = df_out.Memo.str.replace(',', '')
    df_out['AccountType'] = config['account_type']

    return df_out


def load_save(config):
    files = glob.glob(os.path.join(config['folder_in'], '*.csv'))
    print(f"found {len(files)} CSV files in {config['folder_in']}.")
    if len(files) == 0:
        return

    df_list = [load(f, config) for f in files]
    for df_temp in df_list:
        df_temp['count'] = df_temp.groupby(sd.target_columns).cumcount()
    df = pd.concat(df_list)
    df.drop_duplicates().drop(['count'], axis=1).sort_values('Date', ascending=False).to_csv(config['path_out'], index=False)


def load_save_default():
    config = configparser.ConfigParser()
    config.read('config/config.ini')

    load_save(config['Revolut'])
