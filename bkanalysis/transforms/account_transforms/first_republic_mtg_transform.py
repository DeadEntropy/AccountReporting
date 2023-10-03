# coding=utf8
import configparser

import pandas as pd
import glob
import os
from bkanalysis.config.config_helper import parse_list
from bkanalysis.transforms.account_transforms import static_data as sd
from bkanalysis.config import config_helper as ch

from mortgage import Loan


def can_handle(path_in, config):
    if not path_in.lower().endswith('csv'):
        return False
    df = pd.read_csv(path_in, nrows=1)
    expected_columns = parse_list(config['expected_columns'])
    return set(df.columns) == set(expected_columns)

def _postprocess_memo(memos: list, dates: list, amounts: list) -> list:
    idx = 1
    for i in range(len(memos)):
        if memos[i] == "Mobile Deposit":
            memos[i] = f"Mobile Deposit - {dates[i].strftime('%Y%m%d')} - {amounts[i]:.2f}"
            idx = idx + 1

    return memos

def get_payment_breakdown(sorted_dates: list, config) -> pd.DataFrame:
    loan =  Loan(principal=float(config['loan_principal']), interest=float(config['loan_rate']), term=int(config['loan_term']))
    
    interest_payment = {
        d: float(round(sch.interest, 2)) for d, sch in zip(sorted_dates, loan.schedule()[1:len(sorted_dates)+1])
    }

    df_interest = pd.DataFrame(columns=sd.target_columns)
    df_interest.Date = sorted_dates
    df_interest.Account = config['account_name']
    df_interest.Currency = "USD"
    df_interest.Memo = "Mortgage Interest"
    df_interest.Subcategory = "Interest"
    df_interest.Amount = list([-interest_payment[d] for d in sorted_dates])
    
    df_escrow = pd.DataFrame(columns=sd.target_columns)
    df_escrow.Date = sorted_dates
    df_escrow.Account = config['account_name']
    df_escrow.Currency = "USD"
    df_escrow.Memo = "Mortgage Escrow - Flood Insurance"
    df_escrow.Subcategory = "Escrow"
    df_escrow.Amount = -float(config['escrow_amount'])

    return pd.concat([df_interest, df_escrow])

def get_payment_adjustment(df_special: pd.DataFrame):
    df_escrow = df_special[df_special.Memo.str.contains('ESCROW BALANCE')]
    df_escrow['Amount'] = -df_escrow.Amount
    df_escrow['Memo'] = "Mortgage Escrow - Flood Insurance"
    df_escrow['Subcategory'] = "Escrow"
    
    df_interest = df_special[df_special.Memo.str.contains('PREPAID INTEREST')]
    df_interest['Amount'] = -df_interest.Amount
    df_interest['Memo'] = "Mortgage Interest"
    df_interest['Subcategory'] = "Interest"

    return pd.concat([df_interest, df_escrow])

def load(path_in, config):
    df = pd.read_csv(path_in)
    expected_columns = parse_list(config['expected_columns'])
    assert set(df.columns) == set(expected_columns), f'Was expecting [{", ".join(expected_columns)}] but file columns ' \
                                                     f'are [{", ".join(df.columns)}]. (First Republic)'
    

    df_out = pd.DataFrame(columns=sd.target_columns)
    # 
    df_out.Date = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df_out.Account = config['account_name']
    df_out.Currency = "USD"
    memo = list(df['Statement Description'].fillna('Empty Description'))
    df_out.Memo = memo
    subcategory = list(df["Category"].fillna('Empty Category'))
    df_out.Subcategory = subcategory
    df_out['AccountType'] = config['account_type']
    amounts = list(df['Debit'].fillna(0.0) + df['Credit'].fillna(0.0))
    df_out.Amount = amounts
    
    df_payment_breakdown = get_payment_breakdown(list(df_out[df_out.Subcategory == 'Mortgage Payment'].sort_values(by='Date').Date), config)
    df_payment_adjustment = get_payment_adjustment(df_out[df_out.Memo.str.startswith('SPECIAL-PAYMENT')])

    df_out = pd.concat([df_out, df_payment_breakdown, df_payment_adjustment]).sort_values(by='Date', ascending=False)

    assert -df_out.Amount.sum() == df.iloc[0].Balance, 'Balance Mismatch'

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

    load_save(config['FirstRepublic'])
