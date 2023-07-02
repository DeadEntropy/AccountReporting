import configparser
import ast
import pandas as pd
import glob
import os
from bkanalysis.config.config_helper import parse_list
from bkanalysis.transforms.account_transforms import static_data as sd
from bkanalysis.config import config_helper as ch
from bkanalysis.market.market import Market
from bkanalysis.transforms.account_transforms import transformation_helper as helper
import bkanalysis.tax.nutmeg as tax_nut
import re


regex = re.compile('\((.*?)\)')


def can_handle(path_in, config):
    if not path_in.endswith('csv'):
        return False
    df = pd.read_csv(path_in, nrows=1)

    expected_columns = [re.sub(regex, '', s) for s in parse_list(config['expected_columns'])]
    columns = [re.sub(regex, '', s) for s in df.columns]
    return set(columns) == set(expected_columns)


def load(path_in, config):
    df = pd.read_csv(path_in, sep=',', parse_dates=['Date'])
    try:
        expected_columns = [re.sub(regex, '', s) for s in parse_list(config['expected_columns'])]
    except Exception:
        print(config)
        print(config['expected_columns'])
        raise
    columns = [re.sub(regex, '', s) for s in df.columns]
    assert set(columns) == set(expected_columns), f'Was expecting [{", ".join(expected_columns)}] but file columns ' \
                                                     f'are [{", ".join(df.columns)}]. (Nutmeg)'

    df_piv = tax_nut.clean_nutmeg_activity_report(df, include_fund=True)
    df_temp = pd.pivot_table(df_piv,index=['Date', 'Type', 'Fund'], values='Value', aggfunc=sum).fillna(0)

    capital_gain_dfs = []
    for fund in df_piv.Fund.unique():
        df_cap = tax_nut.get_capital_gain_table(df_piv[df_piv.Fund == fund])[['taxable_amount']]
        df_cap['Fund'] = fund
        capital_gain_dfs.append(df_cap)
        
    capital_gain = pd.concat(capital_gain_dfs, axis=0).rename(columns={'taxable_amount': 'Value'})
    capital_gain['Type'] = 'CAP'

    df_temp = df_temp.reset_index().set_index('Date')
    df = pd.concat([df_temp, capital_gain], axis=0).sort_index().reset_index().rename(columns={'index': 'Date'})

    df_out = pd.DataFrame(columns=sd.target_columns)
    df_out.Date = df.Date
    df_out.Account = [f'Nutmeg: {fund[1].strip()}' for fund in df.Fund.str.split(':')]
    df_out.Currency = config['currency']
    df_out.Amount = df.Value
    df_out.Subcategory = df.Type
    df_out.Memo = [f"Nutmeg_{t}" for t in df.Type]
    account_types = ast.literal_eval(config['account_types'])
    df_out['AccountType'] = [account_types[acc] for acc in df_out.Account]

    return df_out


def load_save(config):
    files = glob.glob(os.path.join(config['folder_in'], '*.csv'))
    print(f"found {len(files)} CSV files in {config['folder_in']}.")
    if len(files) == 0:
        return

    df_list = [load(f, config['currency']) for f in files]
    for df_temp in df_list:
        df_temp['count'] = df_temp.groupby(sd.target_columns).cumcount()
    df = pd.concat(df_list)
    df.drop_duplicates().drop(['count'], axis=1).sort_values('Date', ascending=False).to_csv(config['path_out'], index=False)


def load_save_default():
    config = configparser.ConfigParser()
    if len(config.read(ch.source)) != 1:
        raise OSError(f'no config found in {ch.source}')

    load_save(config['Nutmeg'])
