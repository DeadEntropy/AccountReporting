# coding=utf8
import collections
import datetime as dt
import itertools
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from bkanalysis.process import process, status
from bkanalysis.process.iat_identification import IatIdentification
from bkanalysis.projection import projection as pj
from bkanalysis.transforms import master_transform
from bkanalysis.ui.charts.waterfall import plot_waterfall
from bkanalysis.ui.salary import Salary

pd.options.display.float_format = "{:,.0f}".format
logging.basicConfig(level=logging.WARNING)

DATE = "Date"
AMOUNT = "Amount"
AMOUNT_CCY = "AmountCcy"
CUMULATED_AMOUNT = "CumulatedAmount"
CUMULATED_AMOUNT_CCY = "CumulatedAmountCcy"
CAPITAL_GAIN = "CapitalGains"
CUMULATED_AMOUNT_CCY_EXCL_CAPITAL = "CumulatedAmountCcyExclCapGains"
MEMO_MAPPED = "MemoMapped"
TYPE = "Type"
CAPITAL_GAIN = "CapitalGain"
CUMULATED_CAPITAL_GAIN = "CumulatedCapitalGain"


def currency_sign(ccy):
    if ccy == "GBP":
        return "£"
    if ccy == "USD":
        return "$"

    return ccy


def load_transactions(
    save_to_csv=False,
    include_xls=True,
    map_transactions=True,
    config=None,
    include_market=True,
    ignore_overrides=True,
    include_json=True,
    remove_offsetting=True,
):
    mt = master_transform.Loader(config, include_market)
    df_raw = mt.load_all(include_xls, include_json)
    if save_to_csv:
        mt.save(df_raw)

    if not map_transactions:
        return df_raw

    pr = process.Process(config)
    df = pr.process(df_raw, ignore_overrides=ignore_overrides, remove_offsetting=remove_offsetting)
    if save_to_csv:
        pr.save(df)
    pr.__del__()
    return df


def __interpolate(x):
    z = x.ffill().dropna()
    z[AMOUNT] = x[AMOUNT].fillna(0.0)
    z[MEMO_MAPPED] = x[MEMO_MAPPED].fillna("")
    return z


def add_no_capital_column(df):
    df[CAPITAL_GAIN] = [next((float(x[1]) for x in l if x[0] == "CAPITAL"), 0.0) for l in df[MEMO_MAPPED]]
    df[CUMULATED_CAPITAL_GAIN] = df.groupby(["Account", "Currency"])[CAPITAL_GAIN].transform(pd.Series.cumsum)
    df[CUMULATED_AMOUNT_CCY_EXCL_CAPITAL] = df[CUMULATED_AMOUNT_CCY] - df[CUMULATED_CAPITAL_GAIN]


def transactions_to_values(df):
    # Ensure all dates are in the same format
    df.Date = [dt.datetime(old_date.year, old_date.month, old_date.day) for old_date in df.Date]

    # Ensure there all (Account, Currency, Date) tuples are unique
    df.drop(
        df.columns.difference(["Account", "Currency", DATE, MEMO_MAPPED, AMOUNT, "Type"]),
        axis=1,
        inplace=True,
    )
    df = (
        df.groupby(["Account", "Currency", DATE])
        .agg({AMOUNT: ["sum", list], MEMO_MAPPED: list, "Type": list})
        .reset_index()
        .set_index(["Account", "Currency", "Date"])
    )

    df.columns = [" ".join(a) for a in df.columns.to_flat_index()]
    df.rename(
        columns={
            f"{AMOUNT} sum": AMOUNT,
            f"{AMOUNT} list": f"{AMOUNT}Details",
            f"{MEMO_MAPPED} list": MEMO_MAPPED,
            f"{TYPE} list": TYPE,
        },
        inplace=True,
    )
    df[MEMO_MAPPED] = [
        [(i1, i2, t) for (i1, i2, t) in zip(l1, l2, types)] for (l1, l2, types) in zip(df[MEMO_MAPPED], df[f"{AMOUNT}Details"], df[TYPE])
    ]
    df[MEMO_MAPPED] = [memo if memo != "" else {} for memo in df[MEMO_MAPPED]]
    df.drop([f"{AMOUNT}Details", TYPE], axis=1, inplace=True)

    # Compute the running sum for each tuple (Account, Currency)
    df = df.sort_values(DATE)
    df[CUMULATED_AMOUNT] = df.groupby(["Account", "Currency"])[AMOUNT].transform(pd.Series.cumsum)
    df = df.sort_values(DATE, ascending=False)

    # Prepare the Schedule
    date_range = pd.date_range(
        start=df.reset_index().Date.min(),
        end=max(df.reset_index().Date.max(), dt.datetime.now()),
        freq="1D",
    )

    # Create the Index on the full time range
    index = list(
        itertools.product(
            *[
                list(set(list(df.reset_index().set_index(["Account", "Currency"]).index))),
                date_range,
            ]
        )
    )
    index = [(tupl[0], tupl[1], time) for (tupl, time) in index]

    # set the new multi-index
    df = (
        df.reindex(pd.MultiIndex.from_tuples(index, names=df.index.names))
        .reset_index()
        .groupby(["Account", "Currency"])
        .apply(__interpolate)
        .dropna()
        .reset_index(drop=True)
        .set_index(["Account", "Currency"])
    )

    return df


def compute_price(
    df: pd.DataFrame,
    ref_currency: str = "USD",
    period: str = "10y",
    config=None,
    linear_interpolation: bool = False,
    output_market: bool = False,
):
    # Load the market
    from bkanalysis.market import market_loader as ml

    market_loader = ml.MarketLoader(config)
    values = market_loader.load(df.reset_index().Currency.unique(), ref_currency, period)

    # Build the market object
    from bkanalysis.market import market as mkt

    market = mkt.Market(values, linear_interpolation)

    # Compute the value of the each (Account, Currency) in ref_currency
    price_in_currency = [
        (
            market.get_price_in_currency(
                (ml.MarketLoader.get_symbol(instr, ref_currency) if instr not in market_loader.source_map.keys() else instr),
                date,
                ref_currency,
            )
            if instr != ref_currency
            else 1.0
        )
        for (instr, date) in zip(df.reset_index().Currency, df.reset_index().Date)
    ]
    df[CUMULATED_AMOUNT_CCY] = df[CUMULATED_AMOUNT] * price_in_currency
    df[AMOUNT_CCY] = df[AMOUNT] * price_in_currency
    df[MEMO_MAPPED] = [[(k, v * p, t) for (k, v, t) in memo] if memo != "" else [] for (memo, p) in zip(df[MEMO_MAPPED], price_in_currency)]

    cap_series = []
    for idx in df.index.unique():
        cap_series.append(__running_capital_gains(df.loc[idx]))

    df[MEMO_MAPPED] = [memo + [("CAPITAL", c, "C")] if c != 0 else memo for (memo, c) in zip(df[MEMO_MAPPED], pd.concat(cap_series))]
    add_no_capital_column(df)
    if output_market:
        return df, values
    return df


def __running_capital_gains(df):
    assert DATE in df.columns, f"{DATE} not in columns"
    assert CUMULATED_AMOUNT_CCY in df.columns, f"{CUMULATED_AMOUNT_CCY} not in columns"
    assert AMOUNT_CCY in df.columns, f"{AMOUNT_CCY} not in columns"

    df = df.sort_values(DATE)
    return (df[CUMULATED_AMOUNT_CCY].diff() - df[AMOUNT_CCY]).fillna(0)


def add_mappings(df, ref_currency, config=None):
    assert MEMO_MAPPED in df.columns, f"{MEMO_MAPPED} not in columns"

    df_exp = df.explode(MEMO_MAPPED)
    df_exp[AMOUNT_CCY] = [memoMapped[1] if not isinstance(memoMapped, float) else 0 for memoMapped in df_exp[MEMO_MAPPED]]
    df_exp[MEMO_MAPPED] = [memoMapped[0] if not isinstance(memoMapped, float) else "" for memoMapped in df_exp[MEMO_MAPPED]]

    df_exp[MEMO_MAPPED] = df_exp[MEMO_MAPPED].fillna("")

    pr = process.Process(config)
    df_exp = pr.extend_types(df_exp.reset_index(), True, True)
    df_exp["Currency"] = ref_currency

    return pr.remove_offsetting(df_exp, AMOUNT_CCY, False)


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
            raise Exception(f"date_range is not set to a correct value ({date_range}). Expected None or String.")

    df[MEMO_MAPPED] = [memo if memo != "" else {} for memo in df[MEMO_MAPPED]]

    df_on_dates = pd.pivot_table(
        df,
        index="Date",
        values=[by, AMOUNT_CCY, MEMO_MAPPED],
        aggfunc={by: "sum", AMOUNT_CCY: "sum", MEMO_MAPPED: __sum_to_dict},
    )

    values = df_on_dates[by]
    labels = [
        aggregate_memos(memo) + f"<br><br>TOTAL: {d:,.0f}" if d != 0 else ",".join(memo)
        for (memo, d) in zip(df_on_dates[MEMO_MAPPED], df_on_dates[by].diff())
    ]

    return values, labels


def aggregate_memos(memo):
    sorted_memo = dict(sorted(memo.items(), key=lambda item: -abs(item[1])))
    if len(sorted_memo) < 20:
        return "<br>".join([f"{v:>7,.0f}: {k[:25]}" for (k, v) in sorted_memo.items()])
    largest_memos = [
        (k, v) for (k, v) in sorted_memo.items() if abs(v) > sorted([abs(v) for (k, v) in sorted_memo.items()], reverse=True)[19]
    ]
    remainging_amount = sum([v for k, v in sorted_memo.items()]) - sum([v for k, v in largest_memos])
    largest_memos = dict(
        sorted(
            {**dict(largest_memos), **{"OTHER": remainging_amount}}.items(),
            key=lambda item: -abs(item[1]),
        )
    )
    return "<br>".join([f"{v:>7,.0f}: {k[:25]}" for (k, v) in largest_memos.items()])


def get_annotations(df, top_items_count=10):
    l = [
        [[amt, d] + list(memo) for memo in memos if not memo[2] in IatIdentification.iat_types + ["C", "S"]]
        for (amt, memos, d) in zip(df.CumulatedAmountCcyExclCapGains, df.MemoMapped, df.Date)
    ]
    flat_list = [item for sublist in l for item in sublist]
    flat_list.sort(key=lambda x: x[3])
    top_list = flat_list[:top_items_count]

    annotation_data = {}
    for l in top_list:
        if l[1] not in annotation_data:
            annotation_data[l[1]] = (l[0], {l[2]: l[3]})
        elif l[2] in annotation_data[l[1]][1]:
            annotation_data[l[1]][1][l[2]] += l[3]
        else:
            annotation_data[l[1]][1][l[2]] = l[3]

    return annotation_data


def plot_wealth(df, date_range=None, by=CUMULATED_AMOUNT_CCY, top_items_count=0, rename=None):
    if isinstance(by, str):
        values, labels = _get_plot_data(df, date_range, by)
        fig = go.Figure(data=go.Scatter(x=values.index, y=values.values, hovertext=labels))
    elif isinstance(by, list):
        fig = go.Figure()
        for b in by:
            values, labels = _get_plot_data(df, date_range, b)
            fig.add_trace(
                go.Scatter(
                    x=values.index,
                    y=values.values,
                    hovertext=labels,
                    name=b if (rename is None or b not in rename) else rename[b],
                )
            )
    else:
        raise Exception(f"by should be either a float or a list")

    if date_range is None:
        annotations = get_annotations(df, top_items_count)
        values, labels = _get_plot_data(df, date_range, CUMULATED_AMOUNT_CCY_EXCL_CAPITAL)
    else:
        annotations = get_annotations(df[(df.Date > date_range[0]) & (df.Date < date_range[1])], top_items_count)
        values, labels = _get_plot_data(df, date_range, CUMULATED_AMOUNT_CCY_EXCL_CAPITAL)

    fig.update_layout(
        title="Total Wealth",
        xaxis_title="Date",
        yaxis_title="Currency",
        margin=dict(t=50),
    )

    i = 0
    for annotation_date in collections.OrderedDict(sorted(annotations.items())):
        dict_text = annotations[annotation_date][1]
        ann_text = "<br>".join([f"{k[:15]}: {dict_text[k]:,.0f}" for k in dict_text])
        fig.add_annotation(
            x=annotation_date,
            y=values[annotation_date],
            xref="x",
            yref="y",
            text=ann_text,
            showarrow=True,
            font=dict(family="Courier New, monospace", size=12, color="#ffffff"),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=20,
            ay=(-100 if (i % 4) == 0 else 50 if (i % 3) == 0 else -50 if (i % 2) == 0 else 100),
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#ff7f0e",
            opacity=0.65,
        )
        i = i + 1

    return fig


def plot_wealth_yearly(df, by=CUMULATED_AMOUNT_CCY_EXCL_CAPITAL, first_year=2016, last_year=None):
    from datetime import datetime

    import plotly.graph_objects as go

    fig = go.Figure()

    v, l = _get_plot_data(df, by=by)
    labels_r = list(reversed(l))
    i = 0

    fig.update_layout(title="Yearly Wealth", xaxis_title="Date", yaxis_title="Currency")

    if last_year is None:
        last_year = datetime.now().year
    for year in reversed(range(first_year, last_year + 1)):
        if year != last_year:
            color = "grey"
        else:
            color = "blue"
        v_year = v[f"{year}-01-01":f"{year}-12-31"]
        v_year = v_year - v_year.iloc[0]
        v_year.index = v_year.index.strftime("%b-%d")

        lab_year = list(reversed(labels_r[i : i + len(v_year)]))
        i = i + len(v_year)

        fig.add_trace(
            go.Scatter(
                x=v_year.index,
                y=v_year.values,
                mode="lines",
                name=year,
                line_color=color,
                opacity=(year - first_year + 1) / (last_year - first_year + 1),
                hovertext=lab_year,
            )
        )

    fig.show()


def __try_get(d, k, default=None):
    if k in d:
        return d[k]
    elif not (default is None):
        return default
    raise Exception(f"Key {k} is not in dictionary.")


def project(df, nb_years=11, projection_data={}):
    assert CUMULATED_AMOUNT_CCY in df.columns, f'Expect "{CUMULATED_AMOUNT_CCY}" to be in the columns.'
    assert DATE in df.columns, f'Expect "{DATE}" to be in the columns.'

    last_date = df.Date[-1]
    projection = (
        pd.pivot_table(
            df[(df.Date == last_date)],
            values=CUMULATED_AMOUNT_CCY,
            index=["Account"],
            aggfunc="sum",
        )
        .sort_values(CUMULATED_AMOUNT_CCY, ascending=False)
        .reset_index()
    )

    projection["Return"] = [__try_get(projection_data, acc, [0, 0, 0])[0] for acc in projection.Account]
    projection["Volatility"] = [__try_get(projection_data, acc, [0, 0, 0])[1] for acc in projection.Account]
    projection["Contribution"] = [__try_get(projection_data, acc, [0, 0, 0])[2] for acc in projection.Account]

    r = range(0, nb_years)
    (w, w_low, w_up, w_low_ex, w_up_ex) = pj.project_full(projection, r, CUMULATED_AMOUNT_CCY)

    piv = pd.DataFrame(pd.pivot_table(df, values=CUMULATED_AMOUNT_CCY, index=DATE, columns=[], aggfunc="sum").to_records())

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=piv[DATE], y=piv[CUMULATED_AMOUNT_CCY], mode="lines", name="Wealth"))

    projection_x = [pd.Timestamp(piv[DATE].values[-1] + np.timedelta64(365 * i, "D")) for i in r]

    colours = ["#636EFA", "#abbeed", "#bdccf0"]

    fig.add_trace(
        go.Scatter(
            x=projection_x,
            y=w_low_ex,
            fill=None,
            mode="lines",
            name="95% interval",
            line_color=colours[2],
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=projection_x,
            y=w_up_ex,
            fill="tonexty",
            mode="lines",
            name="95% interval",
            line_color=colours[2],
        )
    )

    fig.add_trace(
        go.Scatter(
            x=projection_x,
            y=w_low,
            fill=None,
            mode="lines",
            name="80% interval",
            line_color=colours[1],
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=projection_x,
            y=w_up,
            fill="tonexty",
            mode="lines",
            name="80% interval",
            line_color=colours[1],
        )
    )

    fig.add_trace(
        go.Scatter(
            x=projection_x,
            y=w,
            mode="lines+markers",
            name="expectation",
            line_color=colours[0],
        )
    )

    fig.update_layout(title="Projected Wealth with confidence interval", xaxis_title=DATE)

    return fig


def project_compare(df, nb_years=11, projection_data_1={}, projection_data_2={}):
    assert CUMULATED_AMOUNT_CCY in df.columns, f'Expect "{CUMULATED_AMOUNT_CCY}" to be in the columns.'
    assert DATE in df.columns, f'Expect "{DATE}" to be in the columns.'

    last_date = df.Date[-1]
    projection_1 = (
        pd.pivot_table(
            df[(df.Date == last_date)],
            values=CUMULATED_AMOUNT_CCY,
            index=["Account"],
            aggfunc="sum",
        )
        .sort_values(CUMULATED_AMOUNT_CCY, ascending=False)
        .reset_index()
    )
    projection_1["Return"] = [__try_get(projection_data_1, acc, [0, 0, 0])[0] for acc in projection_1.Account]
    projection_1["Volatility"] = [__try_get(projection_data_1, acc, [0, 0, 0])[1] for acc in projection_1.Account]
    projection_1["Contribution"] = [__try_get(projection_data_1, acc, [0, 0, 0])[2] for acc in projection_1.Account]

    projection_2 = (
        pd.pivot_table(
            df[(df.Date == last_date)],
            values=CUMULATED_AMOUNT_CCY,
            index=["Account"],
            aggfunc="sum",
        )
        .sort_values(CUMULATED_AMOUNT_CCY, ascending=False)
        .reset_index()
    )
    projection_2["Return"] = [__try_get(projection_data_2, acc, [0, 0, 0])[0] for acc in projection_2.Account]
    projection_2["Volatility"] = [__try_get(projection_data_2, acc, [0, 0, 0])[1] for acc in projection_2.Account]
    projection_2["Contribution"] = [__try_get(projection_data_2, acc, [0, 0, 0])[2] for acc in projection_2.Account]

    r = range(0, nb_years)
    (w1, w1_low, w1_up, w1_low_ex, w1_up_ex) = pj.project_full(projection_1, r, CUMULATED_AMOUNT_CCY)
    (w2, w2_low, w2_up, w2_low_ex, w2_up_ex) = pj.project_full(projection_2, r, CUMULATED_AMOUNT_CCY)

    piv = pd.DataFrame(pd.pivot_table(df, values=CUMULATED_AMOUNT_CCY, index=[DATE], columns=[], aggfunc="sum").to_records())

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=piv[DATE], y=piv[CUMULATED_AMOUNT_CCY], mode="lines", name="wealth"))

    projection_x = [pd.Timestamp(piv[DATE].values[-1] + np.timedelta64(365 * i, "D")) for i in r]

    colours1 = ["#636EFA", "#838bfb", "#b5bafd"]

    fig.add_trace(
        go.Scatter(
            x=projection_x,
            y=w1_low_ex,
            fill=None,
            mode="lines",
            name="95% interval",
            line_color=colours1[2],
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=projection_x,
            y=w1_up_ex,
            fill="tonexty",
            mode="lines",
            name="95% interval",
            line_color=colours1[2],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=projection_x,
            y=w1,
            mode="lines+markers",
            name="expectation",
            line_color=colours1[0],
        )
    )

    colours2 = ["#ffb84d", "#ffcc80", "#ffe0b3"]

    fig.add_trace(
        go.Scatter(
            x=projection_x,
            y=w2_low_ex,
            fill=None,
            mode="lines",
            name="95% interval",
            line_color=colours2[2],
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=projection_x,
            y=w2_up_ex,
            fill="tonexty",
            mode="lines",
            name="95% interval",
            line_color=colours2[2],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=projection_x,
            y=w2,
            mode="lines+markers",
            name="expectation",
            line_color=colours2[0],
        )
    )

    fig.update_layout(
        title="Comparison of Projected Wealth with confidence interval",
        xaxis_title=DATE,
    )

    return fig


def get_by(
    df_exp: pd.DataFrame,
    by: str,
    key: str,
    label: str,
    date_range=None,
    nb_days: int = 365,
    show_count: int = 5,
    exclude: list = [],
    max_char: int = 12,
    include_tail_memos: bool = False,
):
    if by not in ["FullMasterType", "FullSubType", "FullType"]:
        raise ValueError(f"by must be in {['FullMasterType', 'FullSubType', 'FullType']}")

    if date_range is None:
        df_exp_date = df_exp[(df_exp.Date > df_exp.Date.max() - timedelta(nb_days))]
    else:
        if len(date_range) == 2:
            df_exp_date = df_exp[(df_exp.Date >= date_range[0]) & (df_exp.Date <= date_range[1])]
        else:
            raise ValueError(f"date_range is not set to a correct value ({date_range}). Expected None or String.")

    if key is None:
        df = pd.pivot_table(
            df_exp_date[(df_exp_date["FullType"] != "Intra-Account Transfert")],
            index=["Date", by, label],
            values=AMOUNT_CCY,
            aggfunc="sum",
        )
    else:
        df = pd.pivot_table(
            df_exp_date[(df_exp_date[by] == key) & (df_exp_date["FullType"] != "Intra-Account Transfert")],
            index=["Date", by, label],
            values=AMOUNT_CCY,
            aggfunc="sum",
        )
    df = pd.DataFrame(df.to_records())

    df["Year"] = [d.year for d in df["Date"]]
    df["Month"] = [f"{d.year}-{d.month}" for d in df["Date"]]
    df = df[df.Month != f"{dt.datetime.now().year}-{dt.datetime.now().month}"]

    if len(df) == 0:
        raise Exception(f"No records found for {key}.")

    df = pd.DataFrame(pd.pivot_table(df, index=["Year", "Month", label], values=AMOUNT_CCY, aggfunc="sum").to_records())
    if len(exclude) > 0:
        df = df[[m not in exclude for m in df[label]]]
    top_memos = list(
        [s for s in pd.pivot_table(df, index=[label], values=AMOUNT_CCY, aggfunc="sum").sort_values(by=AMOUNT_CCY).head(show_count).index]
    )
    if include_tail_memos:
        low_memos = list(
            [
                s
                for s in pd.pivot_table(df, index=[label], values=AMOUNT_CCY, aggfunc="sum")
                .sort_values(by=AMOUNT_CCY)
                .tail(show_count)
                .index
            ]
        )
        top_memos = top_memos + low_memos

    df["Memo"] = [s[:max_char] if s in top_memos else "Other" for s in df[label]]

    df = pd.DataFrame(pd.pivot_table(df, index=["Year", "Month", "Memo"], values=AMOUNT_CCY, aggfunc="sum").to_records()).sort_values(
        by=AMOUNT_CCY
    )
    df[AMOUNT_CCY] = -1 * df[AMOUNT_CCY]
    return df


def plot_spend_waterfall(
    df_exp,
    date_range,
    exclude_fulltypes=["Intra-Account Transfert"],
    exclude_subtypes=[],
    salary_override: Salary = None,
    inc_title=False,
    include_capital_gain=True,
):
    df_exp_filter = df_exp[
        (~df_exp.FullSubType.isin(exclude_subtypes))
        & (~df_exp.FullType.isin(exclude_fulltypes))
        & (df_exp.Date >= date_range[0])
        & (df_exp.Date <= date_range[1])
    ]

    categorised_flows = pd.pivot_table(df_exp_filter, index="FullType", values=AMOUNT_CCY, aggfunc=sum)[AMOUNT_CCY]
    categorised_flows.loc["Pension"] = df_exp_filter[df_exp_filter.FullSubType == "UBS Pension"].AmountCcy.sum()
    if salary_override is not None:
        categorised_flows = categorised_flows.drop("Salary", axis=0)
        for p in salary_override.payrolls:
            categorised_flows.loc[p] = salary_override.actual_salaries[p]

    threshold = 2000
    small_flows = (
        categorised_flows[(categorised_flows > -threshold) & (categorised_flows < threshold) & (categorised_flows.index != "Others")].sum()
        + categorised_flows.loc["Others"]
    )
    categorised_flows = categorised_flows[
        (categorised_flows < -threshold) | (categorised_flows > threshold) | (categorised_flows.index == "Others")
    ]
    categorised_flows.loc["Others"] = small_flows

    if not include_capital_gain:
        categorised_flows = categorised_flows.drop("Capital Gain", axis=0)

    categorised_flows = categorised_flows.sort_values(ascending=False)

    return plot_waterfall(
        categorised_flows,
        (f"Income/Spending Summary {date_range[0]:%b-%y} to {date_range[1]:%b-%y}" if inc_title else None),
    )


def plot_category_breakdown(
    df_exp: pd.DataFrame,
    by: str,
    key: str,
    label: str,
    date_range: list = None,
    nb_days: int = None,
    show_count: int = 5,
    exclude: list = [],
    max_char: int = 12,
    include_tail_memos: bool = False,
):
    df_temp = get_by(
        df_exp,
        by,
        key,
        label,
        date_range,
        nb_days,
        show_count,
        exclude,
        max_char,
        include_tail_memos,
    )
    fig_category_brkdn = px.bar(df_temp, x="Month", y=AMOUNT_CCY, color="Memo", text="Memo")
    fig_category_brkdn.update_layout(title=f"{key} Spending")
    return fig_category_brkdn
