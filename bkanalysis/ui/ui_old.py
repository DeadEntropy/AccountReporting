# coding=utf8
import pandas as pd
import numpy as np

import plotly.express as px

import warnings

pd.options.display.float_format = '{:,.0f}'.format
warnings.filterwarnings("ignore")


def get_reimbursement(df, date_range=None, values='Amount'):
    if date_range is not None:
        df = df[(df.Date > date_range[0]) & (df.Date < date_range[1])]
        if len(df) == 0:
            raise Exception(f'df is empty, check that date_range is correct.')
    else:
        df = df[df.YearToDate < 1]
        if len(df) == 0:
            raise Exception(f'df is empty, check dataframe contains data less than 1 year old.')

    df_expenses = df[(~df.FullMasterType.isin(['Income', 'ExIncome']))
                     & (df.FullType != 'Savings')
                     & (df.FacingAccount == '')
                     & (df[values] > 0)
                     & (df.FullType != 'Intra-Account Transfert')
                     & (df.FullMasterType != 'Savings')
                     & (df.MemoMapped != 'PAYPAL')]
    df_expenses[values] = (-1) * df_expenses[values]

    return df_expenses


def get_expenses(df, date_range=None, values='Amount', inc_reimbursement=False):
    if date_range is not None:
        df = df[(df.Date > date_range[0]) & (df.Date < date_range[1])]
        if len(df) == 0:
            raise Exception(f'df is empty, check that date_range is correct.')
    else:
        df = df[df.YearToDate < 1]
        if len(df) == 0:
            raise Exception(f'df is empty, check dataframe contains data less than 1 year old.')

    df_expenses = df[(~df.FullMasterType.isin(['Income', 'ExIncome']))
                     & (df.FullType != 'Savings')
                     & ((df.FacingAccount == '') | [facc is None for facc in df.FacingAccount])
                     & (df[values] < 0)
                     & (df.FullType != 'Intra-Account Transfert')]
    df_expenses[values] = (-1) * df_expenses[values]

    if inc_reimbursement:
        return pd.concat([df_expenses, get_reimbursement(df, date_range, values)])

    return df_expenses

__DATE_FORMAT = '%Y-%m-%d'


def get_title(dates, amounts):
    try:
        title = f'Spending Breakdown for {np.min(dates).strftime(__DATE_FORMAT)} to ' \
                f'{np.max(dates).strftime(__DATE_FORMAT)}' \
                f' (Total Spend: {amounts.sum():,.0f})'
    except ValueError:
        print(f'{np.min(dates)}')
        print(f'{np.max(dates)}')
        print(f'{amounts.sum()}')
        title = ''

    return title


def plot_sunburst(df, path, date_range=None, values='Amount', inc_reimbursement=False):
    df_expenses = get_expenses(df, date_range, values, inc_reimbursement)
    df_expenses['FullSubType'] = [fullSubType if fullSubType != '' else 'Other' for fullSubType in df_expenses['FullSubType']]
    title = get_title(df_expenses.Date, df_expenses[values])
    sb = px.sunburst(df_expenses, path=path, values=values, title=title)
    return sb
