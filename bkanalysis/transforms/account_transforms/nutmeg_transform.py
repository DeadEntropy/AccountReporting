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


def can_handle(path_in, config, *args):
    if not path_in.endswith('csv'):
        return False
    df = pd.read_csv(path_in, nrows=1)

    expected_columns = [re.sub(regex, '', s) for s in parse_list(config['expected_columns'])]
    columns = [re.sub(regex, '', s) for s in df.columns]
    return set(columns) == set(expected_columns)


def load(path_in, config, *args):
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

    path_in_activity = config['path_activity']
    df = df.rename({'Amount (£)': 'Amount'},axis=1)
    activity = pd.read_csv(path_in_activity,parse_dates=['Date'])\
        .rename({'Share Price (£)': 'Share Price', 'Total Value (£)': 'Total Value'},axis=1)

    dfs = []
    for account in df.Pot.unique():

        acc_trans = df[df.Pot == account].sort_values('Date')
        acc_activ = activity[activity.Pot == account].sort_values('Date')

        df_out = pd.DataFrame(columns=["Date", "Account", "Amount", "Subcategory", "Memo", "Currency"])
        df_out["Date"] = acc_activ['Date']
        df_out["Account"] = f'Nutmeg: {account}'
        df_out["Amount"] = [s if desc =='Purchase' else -s if desc == 'Sale' else -tv if desc == 'Fee' else tv for s,tv,desc in zip(acc_activ['No. Shares'],acc_activ['Total Value'],acc_activ['Description'])] 
        df_out["Currency"] = [asset if desc in ['Purchase', 'Sale'] else 'GBP' for asset,desc in zip(acc_activ['Asset Code'],acc_activ['Description'])] 
        df_out["Subcategory"] = acc_activ['Description']
        df_out["Memo"] = [f'Nutmeg: {account} Fund {desc}' if desc in ['Purchase', 'Sale'] else desc for asset,desc in zip(acc_activ['Asset Code'], acc_activ['Description'])]

        df_out_cash = pd.DataFrame(columns=["Date", "Account", "Amount", "Subcategory", "Memo", "Currency"])
        acc_activ_cash = acc_activ[acc_activ.Description.isin(['Purchase', 'Sale'])]
        df_out_cash["Date"] = acc_activ_cash['Date']
        df_out_cash["Account"] = f'Nutmeg: {account}'
        df_out_cash["Amount"] = [-tv if desc == 'Purchase' else tv for tv,desc in zip(acc_activ_cash['Total Value'], acc_activ_cash['Description'])]
        df_out_cash["Currency"] = 'GBP'
        df_out_cash["Subcategory"] = acc_activ_cash['Description']
        df_out_cash["Memo"] = [f'Nutmeg: {account} Fund {desc}' for asset,desc in zip(acc_activ_cash['Asset Code'], acc_activ_cash['Description'])]

        df_out_trans = pd.DataFrame(columns=["Date", "Account", "Amount", "Subcategory", "Memo", "Currency"])

        df_out_trans["Date"] = acc_trans['Date']
        df_out_trans["Account"] = f'Nutmeg: {account}'
        df_out_trans["Amount"] = acc_trans.Amount
        df_out_trans["Currency"] = 'GBP'
        df_out_trans["Subcategory"] = acc_trans.Description
        df_out_trans["Memo"] = f'Nutmeg: {account}'

        df_out = pd.concat([df_out, df_out_cash, df_out_trans]).sort_values('Date').reset_index(drop=True)
        dfs.append(df_out)
        
    df_out = pd.concat(dfs)

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
