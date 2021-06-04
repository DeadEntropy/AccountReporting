import configparser

import pandas as pd
import glob
import os
from bkanalysis.config.config_helper import parse_list
from bkanalysis.transforms.account_transforms import static_data as sd


def __remove_accents(s):
    return ''.join(' ' if e == "'" else (e if (e.isalnum() or e == ' ') else 'e') for e in s)


def can_handle(path_in, config):
    if not path_in.endswith('xls'):
        return False

    try:
        df = pd.read_excel(path_in)
        df = df.dropna(axis=1, how='all').dropna(axis=0, how='any').reset_index(drop=True)
        df.columns = [__remove_accents(v) for v in list(df.iloc[0])]
    except Exception:
        return False

    expected_columns = parse_list(config['expected_columns'], False)

    return set(df.columns) == set(expected_columns)


def load(path_in, config):
    df = pd.read_excel(path_in)
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='any').reset_index(drop=True)
    df.columns = [__remove_accents(v) for v in list(df.iloc[0])]
    df = df.drop(0)
    expected_columns = parse_list(config['expected_columns'], False)

    assert set(df.columns) == set(expected_columns), \
        f'Was expecting [{", ".join(expected_columns)}] but file columns are [{", ".join(df.columns)}]. (Coinbase)'

    df_out = pd.DataFrame(columns=sd.target_columns)
    df_out.Date = pd.to_datetime(df["Date"])
    df_out.Account = config['account_name']
    df_out.Currency = df["ISIN"]
    df_out.Amount = df["Quantite"]
    df_out.Subcategory = df["Statut"]
    df_out.Memo = [f"BUY: {l}" if q>0 else f"SALE: {l}" for (q, l) in zip(df["Quantite"], df["Libelle"])]
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

    load_save(config['Coinbase'])
