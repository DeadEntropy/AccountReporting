import configparser

import pandas as pd
import glob
import os
from bkanalysis.config.config_helper import parse_list
from bkanalysis.transforms.account_transforms import static_data as sd
from bkanalysis.config import config_helper as ch


def can_handle(path_in, config, sep=',', *args):
    if not path_in.endswith('csv'):
        return False
    df = pd.read_csv(path_in, sep=sep, nrows=1)
    expected_columns = parse_list(config['expected_columns'], False)

    return set(df.columns) == set(expected_columns)


def load(path_in, config, sep=',', *args):
    df = pd.read_csv(path_in, sep=sep)
    df.columns = [s.strip() for s in df.columns]
    expected_columns = parse_list(config['expected_columns'], False)

    assert set(df.columns) == set(expected_columns), \
        f'Was expecting [{", ".join(expected_columns)}] but file columns are [{", ".join(df.columns)}]. (Fidelity)'

    df_out = pd.DataFrame(columns=sd.target_columns)
    df_out.Date = pd.to_datetime(df["Run Date"].str.strip(), format='%m/%d/%Y')
    df_out.Account = 'Fidelity Brokerage'
    df_out.Currency = [config['cash_account'] if s.isspace() else s.strip() for s in df.Symbol]
    df_out.Amount = [a if ccy == config['cash_account'] else q for (q, a, ccy) in
                     zip(df.Quantity, df['Amount ($)'], df_out.Currency)]
    df_out.Memo = df.Action
    df_out.Subcategory = df['Security Description']
    df_out['AccountType'] = config['account_type']

    # Fidelity doesnt give the outflows from the cash_account, so we need to manually add them
    df_cash_account = df[~df.Symbol.str.isspace()]
    df_cash_out = pd.DataFrame(columns=sd.target_columns)
    df_cash_out.Date = pd.to_datetime(df_cash_account["Run Date"].str.strip(), format='%m/%d/%Y')
    df_cash_out.Account = 'Fidelity Brokerage'
    df_cash_out.Currency = config['cash_account']
    df_cash_out.Amount = df_cash_account['Amount ($)']
    df_cash_out.Memo = df_cash_account.Action
    df_cash_out.Subcategory = df_cash_account['Security Description']

    df_out = pd.concat([df_out, df_cash_out]).sort_values('Date', ascending=False).reset_index(drop=True)

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
    if len(config.read(ch.source)) != 1:
        raise OSError(f'no config found in {ch.source}')

    load_save(config['Fidelity'])
