# coding=utf8
import pandas as pd
import numpy as np

import datetime

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
            df[f'Amount_{ref_currency}'] = convert_fx_spot(df, ref_currency)
    else:
        raise Exception(f'{ref_currency} is not a valid ref_currency.')

    df_by = pd.DataFrame(pd.pivot_table(df, values=value_str, index=by, columns=[], aggfunc=sum)
                         .to_records())

    if value_str not in df_by.columns:
        raise Exception(f'{value_str} not in df.columns: {", ".join(df_by.columns)}.')

    df_by = df_by[(df_by[value_str] > 0.01) | (df_by[value_str] < -0.01)]
    df_by = df_by.sort_values(value_str, ascending=False, ignore_index=True)
    return df_by


def convert_fx_spot(df, currency='GBP', key_currency='Currency', key_value='Amount'):
    if isinstance(currency, str):
        df_ccy = df
        fx_spots = mp.get_spot_prices(df_ccy[key_currency].unique(), currency)
        return df_ccy[key_value] * [fx_spots[ccy] for ccy in df_ccy.Currency]
    else:
        raise Exception(f'currency is not set to a correct value ({currency}). Expected None or String.')


def plot_wealth(df, freq='w', currency='GBP', date_range=None, include_internal=False, include_labels=False):
    if include_internal:
        df_ccy = df
    else:
        df_ccy = df[df.FacingAccount == '']

    if date_range is not None:
        if len(date_range) == 2:
            df_ccy = df_ccy[(df_ccy.Date > date_range[0]) & (df_ccy.Date < date_range[1])]
        else:
            raise Exception(f'date_range is not set to a correct value ({date_range}). Expected None or String.')

    if f'Amount_{currency}' not in df_ccy.columns:
        df_ccy[f'Amount_{currency}'] = convert_fx_spot(df_ccy, currency)

    values = df_ccy.set_index('Date').groupby(pd.Grouper(freq=freq))[f'Amount_{currency}'].sum()
    values.update(pd.Series(np.cumsum(values.values), index=values.index))

    if include_labels:
        df_ccy['MemoDetailed'] = df_ccy.MemoMapped + ": " + df_ccy.Amount.map('{:,.0f}'.format)
        df_agg = df_ccy.groupby(['Date', 'MemoDetailed']).agg({f'Amount_{currency}': sum})
        g = df_agg[f'Amount_{currency}'].groupby(level=0, group_keys=False)
        res = g.apply(lambda x: x.sort_values(ascending=False))
        labels = pd.DataFrame(res.to_frame().to_records()).groupby(pd.Grouper(key='Date', freq=freq))[
            'MemoDetailed'].apply(lambda x: '<br>'.join(x.unique()))
        fig = go.Figure(data=go.Scatter(x=values.index, y=values.values, hovertext=labels))
    else:
        fig = go.Figure(data=go.Scatter(x=values.index, y=values.values))

    fig.update_layout(
        title="Total Wealth",
        xaxis_title="Date",
        yaxis_title=currency)

    return fig


def try_get(d, k, default=None):
    if k in d:
        return d[k]
    elif not (default is None):
        return default
    raise Exception(f'Key {k} is not in dictionary.')


def project(df, currency='GBP', nb_years=11, projection_data={}):
    projection = get_current(df, ['AccountType', 'Currency', 'Account']).copy()
    projection['Return'] = [try_get(projection_data, acc, [0, 0, 0])[0] for acc in projection.Account]
    projection['Volatility'] = [try_get(projection_data, acc, [0, 0, 0])[1] for acc in projection.Account]
    projection['Contribution'] = [try_get(projection_data, acc, [0, 0, 0])[2] for acc in projection.Account]

    projection['Amount'] = convert_fx_spot(projection, currency, 'Currency', 'Amount')
    projection['Contribution'] = convert_fx_spot(projection, currency, 'Currency', 'Contribution')

    r = range(0, nb_years)
    (w, w_low, w_up, w_low_ex, w_up_ex) = pj.project_full(projection, r)

    if f'Amount_{currency}' not in df.columns:
        df[f'Amount_{currency}'] = convert_fx_spot(df, currency)

    piv = pd.DataFrame(
        pd.pivot_table(df, values=f'Amount_{currency}', index=['Date'], columns=[], aggfunc=sum).to_records())

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=piv['Date'], y=piv[f'Amount_{currency}'].cumsum(),
                             mode='lines', name='wealth'))

    projection_x = [pd.Timestamp(piv['Date'].values[-1] + np.timedelta64(365 * i, 'D')) for i in r]

    colours = ['#636EFA', '#abbeed', '#bdccf0']

    fig.add_trace(go.Scatter(x=projection_x, y=w_low_ex,
                             fill=None, mode='lines', line_color=colours[2], showlegend=False))
    fig.add_trace(go.Scatter(x=projection_x, y=w_up_ex,
                             fill='tonexty', mode='lines', name='95% interval', line_color=colours[2]))

    fig.add_trace(go.Scatter(x=projection_x, y=w_low,
                             fill=None, mode='lines', line_color=colours[1], showlegend=False))
    fig.add_trace(go.Scatter(x=projection_x, y=w_up,
                             fill='tonexty', mode='lines', name='80% interval', line_color=colours[1]))

    fig.add_trace(go.Scatter(x=projection_x, y=w,
                             mode='lines+markers', name='expectation', line_color=colours[0]))

    fig.update_layout(
        title="Projected Wealth with confidence interval",
        xaxis_title="Date")

    return fig


def get_projection_template(df):
    return {acc: [0, 0, 0] for acc in df.Account.unique()}


def project_compare(df, currency='GBP', nb_years=11, projection_data_1={}, projection_data_2={}):
    projection_1 = get_current(df, ['AccountType', 'Currency', 'Account']).copy()
    projection_1['Return'] = [try_get(projection_data_1, acc, [0, 0, 0])[0] for acc in projection_1.Account]
    projection_1['Volatility'] = [try_get(projection_data_1, acc, [0, 0, 0])[1] for acc in projection_1.Account]
    projection_1['Contribution'] = [try_get(projection_data_1, acc, [0, 0, 0])[2] for acc in projection_1.Account]

    projection_1['Amount'] = convert_fx_spot(projection_1, currency, 'Currency', 'Amount')
    projection_1['Contribution'] = convert_fx_spot(projection_1, currency, 'Currency', 'Contribution')

    projection_2 = get_current(df, ['AccountType', 'Currency', 'Account']).copy()
    projection_2['Return'] = [try_get(projection_data_2, acc, [0, 0, 0])[0] for acc in projection_2.Account]
    projection_2['Volatility'] = [try_get(projection_data_2, acc, [0, 0, 0])[1] for acc in projection_2.Account]
    projection_2['Contribution'] = [try_get(projection_data_2, acc, [0, 0, 0])[2] for acc in projection_2.Account]

    projection_2['Amount'] = convert_fx_spot(projection_2, currency, 'Currency', 'Amount')
    projection_2['Contribution'] = convert_fx_spot(projection_2, currency, 'Currency', 'Contribution')

    r = range(0, nb_years)
    (w1, w1_low, w1_up, w1_low_ex, w1_up_ex) = pj.project_full(projection_1, r)
    (w2, w2_low, w2_up, w2_low_ex, w2_up_ex) = pj.project_full(projection_2, r)

    if f'Amount_{currency}' not in df.columns:
        df[f'Amount_{currency}'] = convert_fx_spot(df, currency)

    piv = pd.DataFrame(
        pd.pivot_table(df, values=f'Amount_{currency}', index=['Date'], columns=[], aggfunc=sum).to_records())

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=piv['Date'], y=piv[f'Amount_{currency}'].cumsum(),
                             mode='lines', name='wealth'))

    projection_x = [pd.Timestamp(piv['Date'].values[-1] + np.timedelta64(365 * i, 'D')) for i in r]

    colours1 = ['#636EFA', '#838bfb', '#b5bafd']

    fig.add_trace(go.Scatter(x=projection_x, y=w1_low_ex,
                             fill=None, mode='lines', line_color=colours1[2], showlegend=False))
    fig.add_trace(go.Scatter(x=projection_x, y=w1_up_ex,
                             fill='tonexty', mode='lines', name='95% interval', line_color=colours1[2]))
    fig.add_trace(go.Scatter(x=projection_x, y=w1,
                             mode='lines+markers', name='expectation', line_color=colours1[0]))

    colours2 = ['#ffb84d', '#ffcc80', '#ffe0b3']

    fig.add_trace(go.Scatter(x=projection_x, y=w2_low_ex,
                             fill=None, mode='lines', line_color=colours2[2], showlegend=False))
    fig.add_trace(go.Scatter(x=projection_x, y=w2_up_ex,
                             fill='tonexty', mode='lines', name='95% interval', line_color=colours2[2]))
    fig.add_trace(go.Scatter(x=projection_x, y=w2,
                             mode='lines+markers', name='expectation', line_color=colours2[0]))

    fig.update_layout(
        title="Comparison of Projected Wealth with confidence interval",
        xaxis_title="Date")

    return fig


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


def get_expenses(df, date_range=None, values='Amount'):
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

    return df_expenses.append(get_reimbursement(df, date_range, values))


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


def plot_sunburst(df, path, date_range=None, values='Amount'):
    df_expenses = get_expenses(df, date_range, values)
    title = get_title(df_expenses.Date, df_expenses[values])
    sb = px.sunburst(df_expenses, path=path, values=values, title=title)
    return sb


def plot_pie(df, index='Account', date_range=None, minimal_amount=1000):
    fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
    plt.close()

    def func(pct, allvals):
        absolute = int(round(pct / 100. * np.sum(allvals)))
        if pct < 1:
            return ''
        return "{:.1f}%\n{:,}".format(pct, absolute)

    df_expenses = get_expenses(df)
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


def compare_expenses(df_trans, year_end=2021):
    df_expenses = {}
    for year in range(year_end - 1, year_end + 1):
        df_expenses[year] = get_expenses(df_trans, date_range=[f'{year}-01-01', f'{year}-12-31'])

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


def capital_gain(df, account, currency, start='2020-12-31', end='2021-12-31'):
    df_1 = df.loc[(account, currency)]
    df_1 = df_1[(df_1.Date > datetime.datetime.strptime(start, "%Y-%m-%d")) & (df_1.Date <= datetime.datetime.strptime(end, "%Y-%m-%d"))]
    df_1 = df_1.sort_values('Date')
    price_end = df_1.CumulatedAmountInCurrency[-1]
    price_start = df_1.CumulatedAmountInCurrency[0] - df_1.AmountInCurrency[0]
    total_invested = df_1.AmountInCurrency.sum()
    price_change = price_end - price_start - total_invested
    return price_change
