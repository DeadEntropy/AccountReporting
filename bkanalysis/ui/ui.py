# coding=utf8
import datetime as dt
import itertools
import warnings
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go
from datetime import timedelta

from bkanalysis.market import market as mkt, market_loader as ml
from bkanalysis.process import process, status
from bkanalysis.transforms import master_transform
from bkanalysis.projection import projection as pj


pd.options.display.float_format = '{:,.0f}'.format
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

DATE = 'Date'
AMOUNT = 'Amount'
AMOUNT_CCY = 'AmountCcy'
CUMULATED_AMOUNT = 'CumulatedAmount'
CUMULATED_AMOUNT_CCY = 'CumulatedAmountCcy'
CAPITAL_GAIN = 'CapitalGains'
CUMULATED_AMOUNT_CCY_EXCL_CAPITAL = 'CumulatedAmountCcyExclCapGains'
MEMO_MAPPED = 'MemoMapped'
TYPE = 'Type'
CAPITAL_GAIN = 'CapitalGain'
CUMULATED_CAPITAL_GAIN = 'CumulatedCapitalGain'


def currency_sign(ccy):
    if ccy == 'GBP':
        return '£'
    if ccy == 'USD':
        return '$'
    
    return ccy


def load_transactions(save_to_csv=False, include_xls=True, map_transactions=True, config=None, include_market=True,
                      ignore_overrides=False):
    mt = master_transform.Loader(config, include_market)
    df_raw = mt.load_all(include_xls)
    if save_to_csv:
        mt.save(df_raw)

    if not map_transactions:
        return df_raw

    pr = process.Process(config)
    df = pr.process(df_raw, ignore_overrides=ignore_overrides)
    if save_to_csv:
        pr.save(df)
    pr.__del__()
    return df


def __interpolate(x):
    z = x.interpolate(method='ffill', limit_direction='forward').dropna()
    z[AMOUNT] = x[AMOUNT].fillna(0.0)
    z[MEMO_MAPPED] = x[MEMO_MAPPED].fillna('')
    return z


def add_no_capital_column(df):
    df[CAPITAL_GAIN] = [next((float(x[1]) for x in l if x[0] == 'CAPITAL'), 0.0) for l in df[MEMO_MAPPED]]
    df[CUMULATED_CAPITAL_GAIN] = df.groupby(['Account', 'Currency'])[CAPITAL_GAIN].transform(pd.Series.cumsum)
    df[CUMULATED_AMOUNT_CCY_EXCL_CAPITAL] = df[CUMULATED_AMOUNT_CCY] - df[CUMULATED_CAPITAL_GAIN]


def transactions_to_values(df):
    # Ensure all dates are in the same format
    df.Date = [dt.datetime(old_date.year, old_date.month, old_date.day) for old_date in df.Date]

    # Ensure there all (Account, Currency, Date) tuples are unique
    df.drop(df.columns.difference(['Account', 'Currency', DATE, MEMO_MAPPED, AMOUNT, 'Type']), 1, inplace=True)
    df = df.groupby(['Account', 'Currency', DATE]).agg(
        {AMOUNT: [sum, list], MEMO_MAPPED: list, 'Type': list}).reset_index().set_index(['Account', 'Currency', 'Date'])

    df.columns = [" ".join(a) for a in df.columns.to_flat_index()]
    df.rename(columns={f'{AMOUNT} sum': AMOUNT,
                       f'{AMOUNT} list': f'{AMOUNT}Details',
                       f'{MEMO_MAPPED} list': MEMO_MAPPED,
                       f'{TYPE} list': TYPE}, inplace=True)
    df[MEMO_MAPPED] = [[(i1, i2, t) for (i1, i2, t) in zip(l1, l2, types)]
                       for (l1, l2, types) in zip(df[MEMO_MAPPED], df[f'{AMOUNT}Details'], df[TYPE])]
    df[MEMO_MAPPED] = [memo if memo != '' else {} for memo in df[MEMO_MAPPED]]
    df.drop([f'{AMOUNT}Details', TYPE], axis=1, inplace=True)

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
    df[CUMULATED_AMOUNT_CCY] = df[CUMULATED_AMOUNT] * price_in_currency
    df[AMOUNT_CCY] = df[AMOUNT] * price_in_currency
    df[MEMO_MAPPED] = [[(k, v * p, t) for (k, v, t) in memo] if memo != '' else []
                       for (memo, p) in zip(
                                            df[MEMO_MAPPED], 
                                            price_in_currency)]

    cap_series = []
    for idx in df.index.unique():
        cap_series.append(__running_capital_gains(df.loc[idx]))

    df[MEMO_MAPPED] = [memo + [('CAPITAL', c, 'C')] if c != 0 else memo for (memo, c) in zip(df[MEMO_MAPPED], pd.concat(cap_series))]
    add_no_capital_column(df)
    return df


def __running_capital_gains(df):
    assert DATE in df.columns, f'{DATE} not in columns'
    assert CUMULATED_AMOUNT_CCY in df.columns, f'{CUMULATED_AMOUNT_CCY} not in columns'
    assert AMOUNT_CCY in df.columns, f'{AMOUNT_CCY} not in columns'

    df = df.sort_values(DATE)
    return (df[CUMULATED_AMOUNT_CCY].diff() - df[AMOUNT_CCY]).fillna(0)


def add_mappings(df, ref_currency, config=None):
    assert MEMO_MAPPED in df.columns, f'{MEMO_MAPPED} not in columns'

    df_exp = df.explode(MEMO_MAPPED)
    df_exp[AMOUNT_CCY] = [memoMapped[1] if not isinstance(memoMapped, float) else 0 for memoMapped in df_exp[MEMO_MAPPED]]
    df_exp[MEMO_MAPPED] = [memoMapped[0] if not isinstance(memoMapped, float) else '' for memoMapped in df_exp[MEMO_MAPPED]]
    
    # Now includes the capital gain
    # assert round(df[AMOUNT_CCY].sum(), 2) == round(df_exp[AMOUNT_CCY].sum(), 2), f'{AMOUNT_CCY} Mismatch'

    df_exp[MEMO_MAPPED] = df_exp[MEMO_MAPPED].fillna('')

    pr = process.Process(config)
    df_exp = pr.extend_types(df_exp.reset_index(), True, True)
    df_exp['Currency'] = ref_currency

    return pr.remove_offsetting(df_exp, False, AMOUNT_CCY, False)


def get_status(df):
    st = status.LastUpdate()
    return st.last_update(df)


# l is a list of tuple with duplicate keys
def __sum_to_dict(l: list):
    out = {}
    for d in l:
        for key in set([e1 for (e1, _, t) in d]):
            if key not in out:
                out[key] = sum([v for (k, v, t) in d if k == key])
            else:
                out[key] += sum([v for (k, v, t) in d if k == key])
                
    return out


def _get_plot_data(df, date_range=None, by=CUMULATED_AMOUNT_CCY):
    assert by in df.columns, f'Expect "{by}" to be in the columns.'
    assert DATE in df.columns, f'Expect "{DATE}" to be in the columns.'
    assert MEMO_MAPPED in df.columns, f'Expect "{MEMO_MAPPED}" to be in the columns.'

    if date_range is not None:
        if len(date_range) == 2:
            df = df[(df.Date > date_range[0]) & (df.Date < date_range[1])]
        else:
            raise Exception(f'date_range is not set to a correct value ({date_range}). Expected None or String.')

    df[MEMO_MAPPED] = [memo if memo != '' else {} for memo in df[MEMO_MAPPED]]
    
    df_on_dates = pd.pivot_table(df, index='Date', values=[by, AMOUNT_CCY, MEMO_MAPPED], 
               aggfunc={by: sum, 
                        AMOUNT_CCY: sum,
                        MEMO_MAPPED:__sum_to_dict})

    values = df_on_dates[by]
    labels = [aggregate_memos(memo) + f"<br><br>TOTAL: {d:,.0f}"\
                if d != 0 else ','.join(memo)\
                for (memo, d) in zip(df_on_dates[MEMO_MAPPED], df_on_dates[by].diff())]

    return values, labels

def aggregate_memos(memo):
    if len(memo) < 20:
        return '<br>'.join([f'{k}: {v:,.0f}' for (k, v) in memo.items()])
    largest_memos = [(k,v) for (k,v) in memo.items() if abs(v) > sorted([abs(v) for (k,v) in memo.items()], reverse=True)[19]]
    remainging_amount = sum([v for k, v in memo.items()]) - sum([v for k, v in largest_memos])
    return '<br>'.join([f'{k}: {v:,.0f}' for (k, v) in largest_memos]) + f'<br>OTHER: {remainging_amount:,.0f}'


def plot_wealth(df, date_range=None, by=CUMULATED_AMOUNT_CCY):
    if isinstance(by, str):
        values, labels = _get_plot_data(df, date_range, by)
        fig = go.Figure(data=go.Scatter(x=values.index, y=values.values, hovertext=labels))
    elif isinstance(by, list):
        fig = go.Figure()
        for b in by:
            values, labels = _get_plot_data(df, date_range, b)
            fig.add_trace(go.Scatter(x=values.index, y=values.values, hovertext=labels, name=b))
    else:
        raise Exception(f'by should be either a float or a list')

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
    assert CUMULATED_AMOUNT_CCY in df.columns, f'Expect "{CUMULATED_AMOUNT_CCY}" to be in the columns.'
    assert DATE in df.columns, f'Expect "{DATE}" to be in the columns.'

    last_date = df.Date[-1]
    projection = pd.pivot_table(df[(df.Date == last_date)], values=CUMULATED_AMOUNT_CCY, index=['Account'], 
                   aggfunc=sum).sort_values(CUMULATED_AMOUNT_CCY, ascending=False).reset_index()

    projection['Return'] = [__try_get(projection_data, acc, [0, 0, 0])[0] for acc in projection.Account]
    projection['Volatility'] = [__try_get(projection_data, acc, [0, 0, 0])[1] for acc in projection.Account]
    projection['Contribution'] = [__try_get(projection_data, acc, [0, 0, 0])[2] for acc in projection.Account]
    
    r = range(0, nb_years)
    (w, w_low, w_up, w_low_ex, w_up_ex) = pj.project_full(projection, r, CUMULATED_AMOUNT_CCY)

    piv = pd.DataFrame(
        pd.pivot_table(df, values=CUMULATED_AMOUNT_CCY, index=DATE, columns=[], aggfunc=sum).to_records())

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=piv[DATE], y=piv[CUMULATED_AMOUNT_CCY],
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
    assert CUMULATED_AMOUNT_CCY in df.columns, f'Expect "{CUMULATED_AMOUNT_CCY}" to be in the columns.'
    assert DATE in df.columns, f'Expect "{DATE}" to be in the columns.'

    last_date = df.Date[-1]
    projection_1 = pd.pivot_table(df[(df.Date == last_date)], values=CUMULATED_AMOUNT_CCY, index=['Account'], 
                   aggfunc=sum).sort_values(CUMULATED_AMOUNT_CCY, ascending=False).reset_index()
    projection_1['Return'] = [__try_get(projection_data_1, acc, [0, 0, 0])[0] for acc in projection_1.Account]
    projection_1['Volatility'] = [__try_get(projection_data_1, acc, [0, 0, 0])[1] for acc in projection_1.Account]
    projection_1['Contribution'] = [__try_get(projection_data_1, acc, [0, 0, 0])[2] for acc in projection_1.Account]


    projection_2 = pd.pivot_table(df[(df.Date == last_date)], values=CUMULATED_AMOUNT_CCY, index=['Account'], 
                   aggfunc=sum).sort_values(CUMULATED_AMOUNT_CCY, ascending=False).reset_index()
    projection_2['Return'] = [__try_get(projection_data_2, acc, [0, 0, 0])[0] for acc in projection_2.Account]
    projection_2['Volatility'] = [__try_get(projection_data_2, acc, [0, 0, 0])[1] for acc in projection_2.Account]
    projection_2['Contribution'] = [__try_get(projection_data_2, acc, [0, 0, 0])[2] for acc in projection_2.Account]


    r = range(0, nb_years)
    (w1, w1_low, w1_up, w1_low_ex, w1_up_ex) = pj.project_full(projection_1, r, CUMULATED_AMOUNT_CCY)
    (w2, w2_low, w2_up, w2_low_ex, w2_up_ex) = pj.project_full(projection_2, r, CUMULATED_AMOUNT_CCY)

    piv = pd.DataFrame(
        pd.pivot_table(df, values=CUMULATED_AMOUNT_CCY, index=[DATE], columns=[], aggfunc=sum).to_records())

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=piv[DATE], y=piv[CUMULATED_AMOUNT_CCY],
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

def get_by_subtype(df_exp:pd.DataFrame, by: str, key: str, nb_days: int = 365, show_count: int = 5, exclude: list = [], max_char: int = 12):
    if by not in ['FullSubType', 'FullType']:
        raise Exception(f"by must be in {['FullSubType', 'FullType']}")
    df = pd.pivot_table(df_exp[(df_exp.Date>df_exp.Date.max()-timedelta(nb_days)) & (df_exp[by] == key)], \
                        index=['Date', 'FullSubType', 'MemoMapped'], values='AmountCcy', aggfunc=sum)
    df = pd.DataFrame(df.to_records())

    df['Year'] = [d.year for d in df['Date']]
    df['Month'] = [f"{d.year}-{d.month}" for d in df['Date']]
    df = df[df.Month != f"{dt.datetime.now().year}-{dt.datetime.now().month}"]
    
    if len(df) == 0:
        raise Exception(f"No records found for {key}.")

    df = pd.DataFrame(pd.pivot_table(df, index=['Year', 'Month', 'MemoMapped'], values='AmountCcy', aggfunc=sum).to_records())
    if len(exclude) > 0:
        df = df[[m not in exclude for m in df.MemoMapped]]
    top_memos = list([s for s in pd.pivot_table(df, index=['MemoMapped'], values='AmountCcy', aggfunc=sum)\
                      .sort_values(by='AmountCcy').head(show_count).index])
    
    df['Memo'] = [s[:max_char] if s in top_memos else 'Other' for s in df.MemoMapped]

    df = pd.DataFrame(pd.pivot_table(df, index=['Year', 'Month', 'Memo'], values='AmountCcy', aggfunc=sum).to_records())\
    .sort_values(by='AmountCcy')
    df['AmountCcy'] = -1 * df['AmountCcy']
    return df