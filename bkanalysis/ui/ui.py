# coding=utf8
import datetime as dt
import itertools
import warnings
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go

from bkanalysis.market import market as mkt, market_loader as ml
from bkanalysis.process import process, status
from bkanalysis.transforms import master_transform
from bkanalysis.projection import projection as pj


pd.options.display.float_format = '{:,.0f}'.format
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

DATE = 'Date'
AMOUNT = 'Amount'
AMOUNT_CURRENCY = 'AmountInCurrency'
CUMULATED_AMOUNT = 'CumulatedAmount'
CUMULATED_AMOUNT_CURRENCY = 'CumulatedAmountInCurrency'
CAPITAL_GAIN = 'CapitalGains'
CUMULATED_AMOUNT_CURRENCY_EXCL_CAPITAL = 'CumulatedAmountInCurrencyExclCapitalGains'
MEMO_MAPPED = 'MemoMapped'


def currency_sign(ccy):
    if ccy == 'GBP':
        return '£'
    if ccy == 'USD':
        return '$'
    
    return ccy


def load_transactions(save_to_csv=False, include_xls=True, map_transactions=True, config=None, include_market=True):
    mt = master_transform.Loader(config, include_market)
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


def __interpolate(x):
    z = x.interpolate(method='ffill', limit_direction='forward').dropna()
    z[AMOUNT] = x[AMOUNT].fillna(0.0)
    z[MEMO_MAPPED] = x[MEMO_MAPPED].fillna('')
    return z


def transactions_to_values(df):
    # Ensure all dates are in the same format
    df.Date = [dt.datetime(old_date.year, old_date.month, old_date.day) for old_date in df.Date]

    # Ensure there all (Account, Currency, Date) tuples are unique
    df.drop(df.columns.difference(['Account', 'Currency', DATE, MEMO_MAPPED, AMOUNT]), 1, inplace=True)
    df = df.groupby(['Account', 'Currency', DATE]).agg(
        {AMOUNT: [sum, list], MEMO_MAPPED: list}).reset_index().set_index(['Account', 'Currency', 'Date'])

    df.columns = [" ".join(a) for a in df.columns.to_flat_index()]
    df.rename(columns={f'{AMOUNT} sum': AMOUNT, f'{AMOUNT} list': f'{AMOUNT}Details', f'{MEMO_MAPPED} list': MEMO_MAPPED}, inplace=True)
    df[MEMO_MAPPED] = [[(i1, i2) for (i1, i2) in zip(l1, l2)] for (l1, l2) in zip(df[MEMO_MAPPED], df[f'{AMOUNT}Details'])]
    df[MEMO_MAPPED] = [memo if memo != '' else {} for memo in df[MEMO_MAPPED]]
    df.drop(f'{AMOUNT}Details', axis=1, inplace=True)

    # Compute the running sum for each tuple (Account, Currency)
    df = df.sort_values(DATE)
    df[CUMULATED_AMOUNT] = df.groupby(['Account', 'Currency'])[AMOUNT].transform(pd.Series.cumsum)
    df = df.sort_values(DATE, ascending=False)

    # Prepare the Schedule
    date_range = pd.date_range(start=df.reset_index().Date.min(),\
                               end=max(df.reset_index().Date.max(), dt.datetime.now()),\
                               freq='1D')

    # Create the Index on the full time range
    index = list(
        itertools.product(*[list(set(list(df.reset_index().set_index(['Account', 'Currency']).index))), date_range]))
    index = [(tupl[0], tupl[1], time) for (tupl, time) in index]

    # set the new multi-index
    df = df.reindex(pd.MultiIndex.from_tuples(index, names=df.index.names)) \
        .reset_index() \
        .groupby(['Account', 'Currency']) \
        .apply(__interpolate) \
        .dropna() \
        .reset_index(drop=True) \
        .set_index(['Account', 'Currency'])

    return df


def compute_price(df: pd.DataFrame, ref_currency: str = 'USD', period: str = '10y', config=None):
    # Load the market
    market_loader = ml.MarketLoader(config)
    values = market_loader.load(df.reset_index().Currency.unique(), ref_currency, period)

    # Build the market object
    market = mkt.Market(values)

    # Compute the value of the each (Account, Currency) in ref_currency
    price_in_currency = [market.get_price_in_currency(
                                ml.MarketLoader.get_symbol(instr, ref_currency) if instr not in market_loader.source_map.keys() else instr, 
                                date, 
                                ref_currency)\
                         if instr != ref_currency else 1.0 
                         for (instr, date) in zip(df.reset_index().Currency, df.reset_index().Date)]
    df[CUMULATED_AMOUNT_CURRENCY] = df[CUMULATED_AMOUNT] * price_in_currency
    df[AMOUNT_CURRENCY] = df[AMOUNT] * price_in_currency
    df[MEMO_MAPPED] = [[(k, v * p) for (k, v) in memo] if memo != '' else {} for (memo, p) in zip(df[MEMO_MAPPED], price_in_currency)]

    return df


def get_status(df):
    st = status.LastUpdate()
    return st.last_update(df)


def capital_gain(df: pd.DataFrame, start=None, end=None):
    assert CUMULATED_AMOUNT_CURRENCY in df.columns, f'Expect "{CUMULATED_AMOUNT_CURRENCY}" to be in the columns.'
    assert AMOUNT_CURRENCY in df.columns, f'Expect "{AMOUNT_CURRENCY}" to be in the columns.'
    assert DATE in df.columns, f'Expect "{DATE}" to be in the columns.'

    if start is not None and end is not None:
        df_range = df[(df.Date > dt.datetime.strptime(start, "%Y-%m-%d")) & (df.Date <= dt.datetime.strptime(end, "%Y-%m-%d"))]
    elif start is not None or end is not None:
        raise Exception(f'start and end must either be both none or both be a date.')

    if len(df_range) == 0:
        return 0

    assert len(df_range.Date) == len(set(df_range.Date)), 'Duplicate Values found in "Date".'

    df_range = df_range.sort_values('Date')
    price_end = list(df_range[CUMULATED_AMOUNT_CURRENCY])[-1]
    price_start = list(df_range[CUMULATED_AMOUNT_CURRENCY])[0] - list(df_range[AMOUNT_CURRENCY])[0]
    total_invested = df_range[AMOUNT_CURRENCY].sum()
    price_change = price_end - price_start - total_invested
    return price_change


# l is a list of tuple with duplicate keys
def __sum_to_dict(l: []):
    out = {}
    for d in l:
        for key in set([e1 for (e1, _) in d]):
            if key not in out:
                out[key] = sum([v for (k, v) in d if k == key])
            else:
                out[key] += sum([v for (k, v) in d if k == key])
                
    return out


def plot_wealth(df, freq='w', date_range=None, include_internal=False, by=CUMULATED_AMOUNT_CURRENCY):
    assert by in df.columns, f'Expect "{by}" to be in the columns.'
    assert DATE in df.columns, f'Expect "{DATE}" to be in the columns.'
    assert MEMO_MAPPED in df.columns, f'Expect "{MEMO_MAPPED}" to be in the columns.'


    if date_range is not None:
        if len(date_range) == 2:
            df = df[(df.Date > date_range[0]) & (df.Date < date_range[1])]
        else:
            raise Exception(f'date_range is not set to a correct value ({date_range}). Expected None or String.')

    df[MEMO_MAPPED] = [memo if memo != '' else {} for memo in df[MEMO_MAPPED]]
    
    df_on_dates = pd.pivot_table(df, index='Date', values=[by, AMOUNT_CURRENCY, MEMO_MAPPED], 
               aggfunc={by: sum, 
                        AMOUNT_CURRENCY: sum,
                        MEMO_MAPPED:__sum_to_dict})

    df2 = df_on_dates.reset_index()
    def capital_gain_so_far(date):
        return capital_gain(df2, start='2013-11-25', end=date)

    df_on_dates[CAPITAL_GAIN] = [capital_gain_so_far(date.strftime('%Y-%m-%d')) for date in df_on_dates.index]

    values = df_on_dates[by]
    labels = ['<br>'.join([f'{k}: {v:,.0f}' for (k, v) in memo.items()]) + f"<br><br>TRANSACT: {sum([v for (k, v) in memo.items()]):,.0f}<br>CAPITAL: {cap:,.0f}<br>TOTAL: {d:,.0f}"\
                if d != 0 else ','.join(memo)\
                for (memo, d, cap) in zip(df_on_dates[MEMO_MAPPED], df_on_dates[by].diff(), df_on_dates[CAPITAL_GAIN].diff())]
    fig = go.Figure(data=go.Scatter(x=values.index, y=values.values, hovertext=labels))
    

    fig.update_layout(
        title="Total Wealth",
        xaxis_title="Date",
        yaxis_title="Currency")

    return fig


def __try_get(d, k, default=None):
    if k in d:
        return d[k]
    elif not (default is None):
        return default
    raise Exception(f'Key {k} is not in dictionary.')


def project(df, nb_years=11, projection_data={}):
    assert CUMULATED_AMOUNT_CURRENCY in df.columns, f'Expect "{CUMULATED_AMOUNT_CURRENCY}" to be in the columns.'
    assert DATE in df.columns, f'Expect "{DATE}" to be in the columns.'

    last_date = df.Date[-1]
    projection = pd.pivot_table(df[(df.Date == last_date)], values=CUMULATED_AMOUNT_CURRENCY, index=['Account'], 
                   aggfunc=sum).sort_values(CUMULATED_AMOUNT_CURRENCY, ascending=False).reset_index()

    projection['Return'] = [__try_get(projection_data, acc, [0, 0, 0])[0] for acc in projection.Account]
    projection['Volatility'] = [__try_get(projection_data, acc, [0, 0, 0])[1] for acc in projection.Account]
    projection['Contribution'] = [__try_get(projection_data, acc, [0, 0, 0])[2] for acc in projection.Account]
    
    r = range(0, nb_years)
    (w, w_low, w_up, w_low_ex, w_up_ex) = pj.project_full(projection, r, CUMULATED_AMOUNT_CURRENCY)

    piv = pd.DataFrame(
        pd.pivot_table(df, values=CUMULATED_AMOUNT_CURRENCY, index=DATE, columns=[], aggfunc=sum).to_records())

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=piv[DATE], y=piv[CUMULATED_AMOUNT_CURRENCY],
                             mode='lines', name='Wealth'))

    projection_x = [pd.Timestamp(piv[DATE].values[-1] + np.timedelta64(365 * i, 'D')) for i in r]

    colours = ['#636EFA', '#abbeed', '#bdccf0']

    fig.add_trace(go.Scatter(x=projection_x, y=w_low_ex,
                             fill=None, mode='lines', name='95% interval', line_color=colours[2], showlegend=False))
    fig.add_trace(go.Scatter(x=projection_x, y=w_up_ex,
                             fill='tonexty', mode='lines', name='95% interval', line_color=colours[2]))

    fig.add_trace(go.Scatter(x=projection_x, y=w_low,
                             fill=None, mode='lines', name='80% interval', line_color=colours[1], showlegend=False))
    fig.add_trace(go.Scatter(x=projection_x, y=w_up,
                             fill='tonexty', mode='lines', name='80% interval', line_color=colours[1]))

    fig.add_trace(go.Scatter(x=projection_x, y=w,
                             mode='lines+markers', name='expectation', line_color=colours[0]))

    fig.update_layout(
        title="Projected Wealth with confidence interval",
        xaxis_title=DATE)

    return fig


def project_compare(df, nb_years=11, projection_data_1={}, projection_data_2={}):
    assert CUMULATED_AMOUNT_CURRENCY in df.columns, f'Expect "{CUMULATED_AMOUNT_CURRENCY}" to be in the columns.'
    assert DATE in df.columns, f'Expect "{DATE}" to be in the columns.'

    last_date = df.Date[-1]
    projection_1 = pd.pivot_table(df[(df.Date == last_date)], values=CUMULATED_AMOUNT_CURRENCY, index=['Account'], 
                   aggfunc=sum).sort_values(CUMULATED_AMOUNT_CURRENCY, ascending=False).reset_index()
    projection_1['Return'] = [__try_get(projection_data_1, acc, [0, 0, 0])[0] for acc in projection_1.Account]
    projection_1['Volatility'] = [__try_get(projection_data_1, acc, [0, 0, 0])[1] for acc in projection_1.Account]
    projection_1['Contribution'] = [__try_get(projection_data_1, acc, [0, 0, 0])[2] for acc in projection_1.Account]


    projection_2 = pd.pivot_table(df[(df.Date == last_date)], values=CUMULATED_AMOUNT_CURRENCY, index=['Account'], 
                   aggfunc=sum).sort_values(CUMULATED_AMOUNT_CURRENCY, ascending=False).reset_index()
    projection_2['Return'] = [__try_get(projection_data_2, acc, [0, 0, 0])[0] for acc in projection_2.Account]
    projection_2['Volatility'] = [__try_get(projection_data_2, acc, [0, 0, 0])[1] for acc in projection_2.Account]
    projection_2['Contribution'] = [__try_get(projection_data_2, acc, [0, 0, 0])[2] for acc in projection_2.Account]


    r = range(0, nb_years)
    (w1, w1_low, w1_up, w1_low_ex, w1_up_ex) = pj.project_full(projection_1, r, CUMULATED_AMOUNT_CURRENCY)
    (w2, w2_low, w2_up, w2_low_ex, w2_up_ex) = pj.project_full(projection_2, r, CUMULATED_AMOUNT_CURRENCY)

    piv = pd.DataFrame(
        pd.pivot_table(df, values=CUMULATED_AMOUNT_CURRENCY, index=[DATE], columns=[], aggfunc=sum).to_records())

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=piv[DATE], y=piv[CUMULATED_AMOUNT_CURRENCY],
                             mode='lines', name='wealth'))

    projection_x = [pd.Timestamp(piv[DATE].values[-1] + np.timedelta64(365 * i, 'D')) for i in r]

    colours1 = ['#636EFA', '#838bfb', '#b5bafd']

    fig.add_trace(go.Scatter(x=projection_x, y=w1_low_ex,
                             fill=None, mode='lines', name='95% interval', line_color=colours1[2], showlegend=False))
    fig.add_trace(go.Scatter(x=projection_x, y=w1_up_ex,
                             fill='tonexty', mode='lines', name='95% interval', line_color=colours1[2]))
    fig.add_trace(go.Scatter(x=projection_x, y=w1,
                             mode='lines+markers', name='expectation', line_color=colours1[0]))

    colours2 = ['#ffb84d', '#ffcc80', '#ffe0b3']

    fig.add_trace(go.Scatter(x=projection_x, y=w2_low_ex,
                             fill=None, mode='lines', name='95% interval', line_color=colours2[2], showlegend=False))
    fig.add_trace(go.Scatter(x=projection_x, y=w2_up_ex,
                             fill='tonexty', mode='lines', name='95% interval', line_color=colours2[2]))
    fig.add_trace(go.Scatter(x=projection_x, y=w2,
                             mode='lines+markers', name='expectation', line_color=colours2[0]))

    fig.update_layout(
        title="Comparison of Projected Wealth with confidence interval",
        xaxis_title=DATE)

    return fig