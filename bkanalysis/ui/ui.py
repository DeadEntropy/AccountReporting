from bkanalysis.process import process, status
from bkanalysis.transforms import master_transform
from bkanalysis.projection import projection as pj
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import numpy as np
import warnings
import plotly.express as px
warnings.filterwarnings("ignore")


def load():
    mt = master_transform.Loader()
    df_raw = mt.load_all()

    pr = process.Process()
    df = pr.process(df_raw)
    pr.__del__()
    return df


def load_csv(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df['FacingAccount'] = df['FacingAccount'].fillna('')
    df['SubType'] = df['SubType'].fillna('')
    df['FullSubType'] = df['FullSubType'].fillna('')
    df['Type'] = df['Type'].fillna('')
    df['FullType'] = df['FullType'].fillna('')
    return df


def get_status(df):
    st = status.LastUpdate()
    return st.last_update(df)


def get_current(df, by=['AccountType', 'Currency']):
    df_by = pd.DataFrame(pd.pivot_table(df, values='Amount', index=by, columns=[], aggfunc=sum)
                         .to_records())

    df_by = df_by[(df_by.Amount > 1) | (df_by.Amount < -1)].sort_values('Amount', ascending=False, ignore_index=True)
    return df_by


def plot_wealth_1(df):
    piv_ccy = pd.DataFrame(
        pd.pivot_table(df, values='Amount', index=['Currency', 'Date'], columns=[], aggfunc=sum).to_records())

    fig, ax = plt.subplots(figsize=(20, 10))
    for ccy in piv_ccy.Currency.unique():
        ax.plot(piv_ccy[piv_ccy.Currency == ccy]['Date'], piv_ccy[piv_ccy.Currency == ccy]['Amount'].cumsum(),
                label=ccy)

    ax.legend()
    ax.grid()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%Y-%m"))


def plot_wealth_2(df, freq='w', currency=None, date_range=None):

    if date_range is None:
        df_date_range = df
    elif len(date_range) == 2:
        df_date_range = df[(df.Date>date_range[0]) & (df.Date<date_range[1])]
    else:
        raise Exception(f'date_range is not set to a correct value ({date_range}). Expected None or String.')

    if currency is None:
        df_ccy = df_date_range
    elif isinstance(currency, str):
        df_ccy = df_date_range[df_date_range.Currency == currency]
    else:
        raise Exception(f'currency is not set to a correct value ({currency}). Expected None or String.')

    values = df_ccy.set_index('Date').groupby(pd.Grouper(freq=freq))['Amount'].sum()
    values.update(pd.Series(np.cumsum(values.values), index=values.index))
    df_ccy['MemoDetailed'] = df_ccy.MemoMapped + ": " + df_ccy.Amount.map('{:,.0f}'.format)
    df_agg = df_ccy[df_ccy.FacingAccount == ''].groupby(['Date', 'MemoDetailed']).agg({'Amount': sum})
    g = df_agg['Amount'].groupby(level=0, group_keys=False)
    res = g.apply(lambda x: x.sort_values(ascending=False))
    labels = pd.DataFrame(res.to_frame().to_records()).groupby(pd.Grouper(key='Date', freq=freq))[
        'MemoDetailed'].apply(lambda x: '<br>'.join(x.unique()))

    fig = go.Figure(data=go.Scatter(x=values.index, y=values.values, hovertext=labels))

    fig.update_layout(
        title="Total Wealth",
        xaxis_title="Date")

    return fig


def project(df, nb_years=11, projection_data='account_wealth_projection.csv'):
    projection = pd.read_csv(projection_data)
    r = range(0, nb_years)
    (w, w_low, w_up, w_low_ex, w_up_ex) = pj.project_full(projection, r)

    fig, ax = plt.subplots(figsize=(15, 7))
    piv = pd.DataFrame(pd.pivot_table(df, values='Amount', index=['Date'], columns=[], aggfunc=sum).to_records())

    ax.plot(piv['Date'], piv['Amount'].cumsum())
    ax.plot([piv['Date'].values[-1] + np.timedelta64(365 * i, 'D') for i in r], w, '-o', color='b',
            label='Expected Wealth')
    ax.fill_between([piv['Date'].values[-1] + np.timedelta64(365 * i, 'D') for i in r], w_low, w_up, color='b',
                    alpha=.2, label='Less Likely')
    ax.fill_between([piv['Date'].values[-1] + np.timedelta64(365 * i, 'D') for i in r], w_low_ex, w_up_ex, color='b',
                    alpha=.1, label='Most Likely')

    ax.grid()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%Y-%m"))


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


def plot_sunburst(df, path, account=None, currency=None, date_range=None):

    if date_range is not None:
        df = df[(df.Date > date_range[0]) & (df.Date < date_range[1])]
    else:
        df = df[df.YearToDate < 1]

    if currency is not None:
        df = df[df.Currency == currency]
    if account is not None:
        df = df[df.Account == account]

    df_expenses = df[(df.FullType != 'Savings')
                     & (df.FacingAccount == '')
                     & (df.Amount < 0)
                     & (df.FullType != 'Intra-Account Transfert')]
    df_expenses.Amount = (-1) * df_expenses.Amount

    fig = px.sunburst(df_expenses, path=path, values='Amount')
    return fig
