# coding=utf8
import pandas as pd
import numpy as np


from bkanalysis.process import process, status
from bkanalysis.transforms import master_transform
from bkanalysis.projection import projection as pj
from bkanalysis.market import market_prices as mp

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

import warnings

pd.options.display.float_format = '{:,.0f}'.format
warnings.filterwarnings("ignore")


def load(save_to_csv=False, include_xls=True, map_transactions=True, config=None):
    mt = master_transform.Loader(config)
    df_raw = mt.load_all(include_xls)
    if save_to_csv:
        mt.save(df_raw)

    if not map_transactions:
        return df_raw

    pr = process.Process(config)
    df = pr.process(df_raw)
    if save_to_csv:
        pr.save(df)
    pr.__del__()
    return df


def get_status(df):
    st = status.LastUpdate()
    return st.last_update(df)


def get_current(df, by=['AccountType', 'Currency'], ref_currency=None):
    if ref_currency is None:
        value_str = 'Amount'
    elif len(ref_currency) == 3:
        value_str = f'Amount_{ref_currency}'
        if value_str not in df.columns:
            df[f'Amount_{ref_currency}'] = None # convert_fx_spot(df, ref_currency)
    else:
        raise Exception(f'{ref_currency} is not a valid ref_currency.')

    df_by = pd.DataFrame(pd.pivot_table(df, values=value_str, index=by, columns=[], aggfunc=sum)
                         .to_records())

    if value_str not in df_by.columns:
        raise Exception(f'{value_str} not in df.columns: {", ".join(df_by.columns)}.')

    df_by = df_by[(df_by[value_str] > 0.01) | (df_by[value_str] < -0.01)]
    df_by = df_by.sort_values(value_str, ascending=False, ignore_index=True)
    return df_by


def internal_flows(df):
    df_internal = df[(df.FacingAccount != '') & (df.Amount > 1000) & (df.YearToDate < 5)]
    df_internal = pd.DataFrame(
        pd.pivot_table(df_internal, values='Amount', index=['Account', 'FacingAccount'], columns=[],
                       aggfunc=sum).to_records())
    map_list = list(set(sorted(list(set(df_internal.Account.values)) + list(set(df_internal.FacingAccount.values)))))

    fig = go.Figure(data=[go.Sankey(
        valueformat=".0f",
        valuesuffix="£",
        # Define nodes
        node=dict(
            pad=15,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=map_list,
        ),
        # Add links
        link=dict(
            source=[map_list.index(s) for s in df_internal['Account'].values],
            target=[map_list.index(s) for s in df_internal['FacingAccount'].values],
            value=df_internal['Amount']
        ))])

    fig.update_layout(font_size=10)
    return fig


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
                     & (df.FacingAccount == '')
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
                f' (Total Spend: {np.abs(amounts.sum()):,.0f})'
    except ValueError:
        print(f'{np.min(dates)}')
        print(f'{np.max(dates)}')
        print(f'{np.abs(amounts.sum())}')
        title = ''

    return title


def plot_sunburst(df, path, date_range=None, values='Amount', inc_reimbursement=False):
    df_expenses = get_expenses(df, date_range, values, inc_reimbursement)
    df_expenses['FullSubType'] = [fullSubType if fullSubType != '' else 'Other' for fullSubType in df_expenses['FullSubType']]
    title = get_title(df_expenses.Date, df_expenses[values])
    sb = px.sunburst(df_expenses, path=path, values=values, title=title)
    return sb


def plot_pie(df, index='Account', date_range=None, minimal_amount=1000, inc_reimbursement=False):
    fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
    plt.close()

    def func(pct, allvals):
        absolute = int(round(pct / 100. * np.sum(allvals)))
        if pct < 1:
            return ''
        return "{:.1f}%\n{:,}".format(pct, absolute)

    df_expenses = get_expenses(df, date_range, inc_reimbursement=inc_reimbursement)
    df_expenses["Amount"] = pd.to_numeric(df_expenses["Amount"])
    df_account = pd.DataFrame(pd.pivot_table(df_expenses, values=['Amount'], index=index, aggfunc=sum).to_records())
    df_account = df_account[df_account.Amount > minimal_amount]
    df_account = df_account.sort_values('Amount', ascending=False)

    account = list(df_account[index])
    amount = list(df_account.Amount)

    wedges, texts, autotexts = ax.pie(amount, labels=account, autopct=lambda pct: func(pct, amount),
                                      textprops=dict(color="w"))

    # plt.setp(autotexts, size=12, weight="bold")

    title = get_title(df_expenses.Date, df_expenses.Amount)
    ax.set_title(title)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)

    return fig


def plot_expenses_pie(df, full_type=None, values='Amount', year_end=2021):
    sunbursts = {}
    for year in range(year_end - 1, year_end+1):
        if full_type is None:
            sunbursts[year] = plot_sunburst(df, ['FullType', 'FullSubType'],
                                            date_range=[f'{year}-01-01', f'{year}-12-31'], values=values)
        else:
            sunbursts[year] = plot_sunburst(df[df.FullType == full_type], ['FullSubType','MemoMapped'],
                                            date_range=[f'{year}-01-01', f'{year}-12-31'], values=values)

    fig = make_subplots(rows=1, cols=2, specs=[
        [{"type": "sunburst"}, {"type": "sunburst"}]
    ], subplot_titles=(sunbursts[year_end - 1].layout['title']['text'], sunbursts[year].layout['title']['text']))

    fig.add_trace(sunbursts[year_end - 1].data[0], row=1, col=1)
    fig.add_trace(sunbursts[year_end].data[0], row=1, col=2)
    return fig


def compare_expenses(df_trans, year_end=2021, inc_reimbursement=False):
    df_expenses = {}
    for year in range(year_end - 1, year_end + 1):
        df_expenses[year] = get_expenses(df_trans, date_range=[f'{year}-01-01', f'{year}-12-31'], inc_reimbursement=inc_reimbursement)

    labels = list(set(df_expenses[year_end - 1].FullType) | set(df_expenses[year_end].FullType))
    values_2020 = [(-1) * df_expenses[year_end - 1][df_expenses[year_end - 1].FullType == label].Amount_USD.sum() for
                   label in labels]
    values_2021 = [(-1) * df_expenses[year_end][df_expenses[year_end].FullType == label].Amount_USD.sum() for label in
                   labels]
    values_diff = [(-1) * df_expenses[year_end][df_expenses[year_end].FullType == label].Amount_USD.sum() - (-1) *
                   df_expenses[year_end - 1][df_expenses[year_end - 1].FullType == label].Amount_USD.sum() for label in
                   labels]

    df = pd.DataFrame({'Label': labels, 'Values Diff': values_diff, f'Values {year_end - 1}': values_2020,
                       f'Values {year_end}': values_2021})
    df = df.sort_values('Values Diff')

    return df


def plot_expenses_bar(df, year_end=2021, figsize=(12, 6), dpi=320):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    x = np.arange(df.shape[0])  # the label locations
    width = 0.35  # the width of the bars

    rects_diff = ax.bar(x, df['Values Diff'], width, label=f'{year_end - 1} vs {year_end}')

    ax.set_title(f'Change in expenses by Category between {year_end - 1} and {year_end}')
    ax.set_xticks(x, df.Label, rotation=45)
    ax.grid()
    ax.bar_label(rects_diff, padding=3, rotation=45, labels=[f'{v.get_height():,.0f}' for v in rects_diff], fontsize=7)
    fig.tight_layout()


def plot_budget_bar(df, year_end=2021):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=320)

    x = np.arange(df.shape[0])  # the label locations
    width = 0.35  # the width of the bars

    rects_budget = ax.bar(x + width * 3 / 4, df[f'Budget {year_end + 1}'], width, label=f'{year_end + 1} Budget')
    ax.bar(x - width / 2, df[f'Values {year_end - 1}'], width / 2, label=f'{year_end - 1} Spending', alpha=0.5)
    ax.bar(x, df[f'Values {year_end}'], width / 2, label=f'{year_end} Spending', alpha=0.5)

    ax.set_title(f'Budget for {year_end + 1}')
    ax.set_xticks(x, df.index, rotation=45)
    ax.grid()
    ax.legend()
    ax.bar_label(rects_budget, padding=3, rotation=45, labels=[f'{v.get_height():,.0f}' for v in rects_budget],
                 fontsize=7)
    fig.tight_layout()
    
