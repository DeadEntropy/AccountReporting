# coding=utf8
from bkanalysis.process import process, status
from bkanalysis.transforms import master_transform
from bkanalysis.projection import projection as pj
from bkanalysis.market import market_prices as mp
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import warnings
import plotly.express as px


pd.options.display.float_format = '{:,.0f}'.format
warnings.filterwarnings("ignore")


def load(save_to_csv=False, include_xls=True):
    mt = master_transform.Loader()
    df_raw = mt.load_all(include_xls)
    if save_to_csv:
        mt.save(df_raw)

    pr = process.Process()
    df = pr.process(df_raw)
    if save_to_csv:
        pr.save(df)
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


def get_current(df, by=['AccountType', 'Currency'], ref_currency=None):
    if ref_currency is None:
        value_str = 'Amount'
    elif len(ref_currency) == 3:
        value_str = f'Amount_{ref_currency}'
        if value_str not in df.columns:
            df[f'Amount_{ref_currency}'] = convert_fx(df, ref_currency)
    else:
        raise Exception(f'{ref_currency} is not a valid ref_currency.')

    df_by = pd.DataFrame(pd.pivot_table(df, values=value_str, index=by, columns=[], aggfunc=sum)
                         .to_records())

    df_by = df_by[(df_by[value_str] > 0.01) | (df_by[value_str] < -0.01)].sort_values(value_str, ascending=False, ignore_index=True)
    return df_by


def convert_fx(df, currency='GBP', key_currency='Currency', key_value='Amount'):
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
        df_ccy[f'Amount_{currency}'] = convert_fx(df_ccy, currency)

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

    projection['Amount'] = convert_fx(projection, currency, 'Currency', 'Amount')
    projection['Contribution'] = convert_fx(projection, currency, 'Currency', 'Contribution')

    r = range(0, nb_years)
    (w, w_low, w_up, w_low_ex, w_up_ex) = pj.project_full(projection, r)

    if f'Amount_{currency}' not in df.columns:
        df[f'Amount_{currency}'] = convert_fx(df, currency)

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

    projection_1['Amount'] = convert_fx(projection_1, currency, 'Currency', 'Amount')
    projection_1['Contribution'] = convert_fx(projection_1, currency, 'Currency', 'Contribution')

    projection_2 = get_current(df, ['AccountType', 'Currency', 'Account']).copy()
    projection_2['Return'] = [try_get(projection_data_2, acc, [0, 0, 0])[0] for acc in projection_2.Account]
    projection_2['Volatility'] = [try_get(projection_data_2, acc, [0, 0, 0])[1] for acc in projection_2.Account]
    projection_2['Contribution'] = [try_get(projection_data_2, acc, [0, 0, 0])[2] for acc in projection_2.Account]

    projection_2['Amount'] = convert_fx(projection_2, currency, 'Currency', 'Amount')
    projection_2['Contribution'] = convert_fx(projection_2, currency, 'Currency', 'Contribution')

    r = range(0, nb_years)
    (w1, w1_low, w1_up, w1_low_ex, w1_up_ex) = pj.project_full(projection_1, r)
    (w2, w2_low, w2_up, w2_low_ex, w2_up_ex) = pj.project_full(projection_2, r)

    if f'Amount_{currency}' not in df.columns:
        df[f'Amount_{currency}'] = convert_fx(df, currency)

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


__DATE_FORMAT = '%Y-%m-%d'


def plot_sunburst(df, path, account=None, currency=None, date_range=None):
    if date_range is not None:
        df = df[(df.Date > date_range[0]) & (df.Date < date_range[1])]
    else:
        df = df[df.YearToDate < 1]

    if currency is not None:
        df = df[df.Currency == currency]
    if account is not None:
        df = df[df.Account == account]

    df_expenses = df[(~df.FullMasterType.isin(['Income', 'ExIncome']))
                     & (df.FullType != 'Savings')
                     & (df.FacingAccount == '')
                     & (df.Amount < 0)
                     & (df.FullType != 'Intra-Account Transfert')]
    df_expenses.Amount = (-1) * df_expenses.Amount

    title = f'Spending Breakdown for {np.min(df.Date).strftime(__DATE_FORMAT)} to {np.max(df.Date).strftime(__DATE_FORMAT)}' \
            f' (Total Spend: {np.abs(df_expenses.Amount.sum()):,.0f})'
    fig = px.sunburst(df_expenses, path=path, values='Amount', title=title)
    return fig
