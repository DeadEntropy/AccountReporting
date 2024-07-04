# coding=utf8
import configparser

import pandas as pd
import glob
import os
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta

from mortgage import Loan

from bkanalysis.transforms.account_transforms import static_data as sd
from bkanalysis.config import config_helper as ch

### There is an inacuracy inthe handling of interest with overpayment

def can_handle(path_in, config, *args):
    if not path_in.lower().endswith('json'):
        return False
    
    if os.path.exists(path_in):
        try:
            with open(path_in, 'r') as json_file:
                json_obj = json.load(json_file)
        except TypeError:
            raise TypeError(f"Failed to deserialise jSon file '{path_in}' {json_file}")
    else:
        raise TypeError(f"Failed to local jSon file '{path_in}' {json_file}")
    
    return 'interest' in json_obj[list(json_obj.keys())[0]]


def load(path_in: str, config, *args):
    if os.path.exists(path_in):
        try:
            with open(path_in, 'r') as json_file:
                json_obj = json.load(json_file)
        except TypeError:
            raise TypeError(f"Failed to deserialise jSon file '{path_in}' {json_file}")
    else:
        raise TypeError(f"Failed to local jSon file '{path_in}' {json_file}")
    
    dfs = []
    for _, data in json_obj.items():
        dfs.append(get_cashflows(data))        
        
    return pd.concat(dfs).sort_values('Date')

def get_cashflows(data):
    start_date = datetime.strptime(data['start_date'], '%d-%b-%Y')
    interest = data['interest']
    term = data['term']
    principal = data['principal']
    escrow = data['escrow'] if 'escrow' in data else 0 
    if escrow < 0:
        raise ValueError(f'Mortgage escrow cannot be negative: {escrow}.')
    account_name = data['account_name']
    currency = data['currency']
    include_balance = data['include_balance'] if "include_balance" in data else False
    overpayment = data['overpayment'] if 'overpayment' in data else 0
    if overpayment < 0:
        raise ValueError(f'Mortgage Overpayment cannot be negative: {overpayment}.')
    payments_per_month = data['payments_per_month'] if 'payments_per_month' in data else 1
    
    days_alive = (datetime.now() - start_date).days
    months_alive = int(days_alive / (365.25/12))

    skip_periods = data['skip_periods'] + 1 if 'skip_periods' in data else 1
    take_periods = data['take_periods'] if 'take_periods' in data else months_alive + 1
    last_period = min(months_alive + 1, skip_periods + take_periods)

    loan = Loan(principal=principal, interest=interest, term=term)

    dates = [start_date + relativedelta(months=i) for i in range(skip_periods, last_period)]
    payments = [float(l.payment) + escrow + overpayment for l in loan.schedule()[skip_periods:last_period]]
    interests = [-float(l.interest) for l in loan.schedule()[skip_periods:last_period]]

    if payments_per_month > 1:
        dates = [x for y in [[d, d + relativedelta(weeks=2)] for d in dates] for x in y]
        payments = [x for y in [[p/2, p/2] for p in payments] for x in y]
        interests = [x for y in [[i/2, i/2] for i in interests] for x in y]

    df_balance = pd.DataFrame(columns=['Date','Account', 'Currency','Amount','Subcategory','Memo', 'AccountType'])
    if include_balance:
        balance = loan.schedule()[skip_periods].balance
        df_balance.Date = [dates[0]]
        df_balance.Account = account_name
        df_balance.Currency = currency
        df_balance.Amount = -round(float(balance),2)
        df_balance.Subcategory = 'Transfer'
        df_balance.Memo = 'Loan Transfer'
        df_balance['AccountType'] = 'mortgage'
    
    df_payment = pd.DataFrame(columns=['Date','Account', 'Currency','Amount','Subcategory','Memo', 'AccountType'])
    df_payment.Date = dates
    df_payment.Account = account_name
    df_payment.Currency = currency
    df_payment.Amount = payments
    df_payment.Subcategory = 'Payments'
    df_payment.Memo = 'Loan Payment'
    df_payment['AccountType'] = 'mortgage'

    df_interest = pd.DataFrame(columns=['Date','Account', 'Currency','Amount','Subcategory','Memo', 'AccountType'])
    df_interest.Date = dates
    df_interest.Account = account_name
    df_interest.Currency = currency
    df_interest.Amount = interests
    df_interest.Subcategory = 'Interest'
    df_interest.Memo = 'Mortgage Interest'
    df_interest['AccountType'] = 'mortgage'

    df_escrow = pd.DataFrame(columns=['Date','Account', 'Currency','Amount','Subcategory','Memo', 'AccountType'])
    if escrow != 0:
        df_escrow.Date = dates
        df_escrow.Account = account_name
        df_escrow.Currency = currency
        df_escrow.Amount = -escrow
        df_escrow.Subcategory = 'Escrow'
        df_escrow.Memo = 'Mortgage Escrow - Flood Insurance'
        df_escrow['AccountType'] = 'mortgage'

    df_mtg = pd.concat([df_balance, df_payment, df_interest, df_escrow], axis=0)
    return df_mtg[df_mtg.Date <= datetime.now()]

def load_save(config):
    files = glob.glob(os.path.join(config['folder_in'], '*.json'))
    print(f"found {len(files)} JSON files in {config['folder_in']}.")
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

    load_save(config['Script'])
