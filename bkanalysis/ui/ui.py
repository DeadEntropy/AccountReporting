# coding=utf8
import datetime as dt
import itertools
import warnings
import pandas as pd

from bkanalysis.market import market as mkt, market_loader as ml
from bkanalysis.process import process, status
from bkanalysis.transforms import master_transform

pd.options.display.float_format = '{:,.0f}'.format
warnings.filterwarnings("ignore")


def load_transactions(save_to_csv=False, include_xls=True, map_transactions=True, config=None):
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


def transactions_to_values(df):
    # Ensure all dates are in the same format
    df.Date = [dt.datetime(old_date.year, old_date.month, old_date.day) for old_date in df.Date]

    # Ensure there all (Account, Currency, Date) tuples are unique
    df.drop(df.columns.difference(['Account', 'Currency', 'Date', 'MemoMapped', 'Amount']), 1, inplace=True)
    df = df.groupby(['Account', 'Currency', 'Date']).agg(
        {'Amount': 'sum', 'MemoMapped': sum}).reset_index().drop_duplicates().set_index(['Account', 'Currency', 'Date'])

    # Compute the running sum for each tuple (Account, Currency)
    df = df.sort_values('Date')
    df['CumulatedAmount'] = df.groupby(['Account', 'Currency'])['Amount'].transform(pd.Series.cumsum)
    df = df.sort_values('Date', ascending=False)

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
        .apply(lambda group: group.interpolate(method='ffill', limit_direction='forward')) \
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
    price_in_currency = [market.get_price_in_currency(ml.MarketLoader.get_symbol(instr, ref_currency), date, ref_currency)\
                         if instr != ref_currency else 1.0 for (instr, date) in zip(df.reset_index().Currency, df.reset_index().Date)]
    df['CumulatedAmountInCurrency'] = df['CumulatedAmount'] * price_in_currency

    return df


def get_status(df):
    st = status.LastUpdate()
    return st.last_update(df)
