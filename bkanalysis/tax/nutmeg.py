import pandas as pd
import numpy as np

__EPSILON = 0.01


def clean_nutmeg_activity_report(df_activity, fund_list: list = None, include_fund: bool = False):
    if fund_list is None:
        fund_list = ["Djokovic, Nicolas : Rainy day pot", "Djokovic, Nicolas : My ISA"]
    df = df_activity[df_activity.Fund.isin(fund_list)]

    df.Value = df.Value.str.replace(",", "").astype("float")
    df.Units = df.Units.str.replace(",", "").astype("float")
    df["Units(S)"] = df["Units(S)"].str.replace(",", "").astype("float")
    df["Value(S)"] = df["Value(S)"].str.replace(",", "").astype("float")

    df["Unitary Value"] = [v / u if u != 0.0 else 0.0 for (u, v) in zip(df["Units(S)"], df["Value(S)"])]
    df = df.set_index("Date", drop=True).sort_index()

    index = ["Date", "Asset Code", "Type", "Narrative"]
    if include_fund:
        index = index + ["Fund"]
    df_piv = pd.DataFrame(
        pd.pivot_table(
            df,
            index=index,
            values=["Units(S)", "Value(S)", "Unitary Value"],
            aggfunc={"Units(S)": sum, "Value(S)": sum, "Unitary Value": np.mean},
        ).to_records()
    ).sort_values("Date")
    df_piv = df_piv.rename({"Units(S)": "Units", "Value(S)": "Value"}, axis=1)
    df_piv = df_piv.set_index("Date", drop=True)

    return df_piv


def re_sign_values(x, t):
    if t == "SLD":
        return -x
    if t == "FEE":
        return -x
    return x


def clean_nutmeg_investment_activity(df_investment, fund_list: list = None, include_fund: bool = False):
    if fund_list is None:
        fund_list = ["Rainy day pot", "My ISA"]

    df = df_investment[df_investment.Pot.isin(fund_list)]
    df = df.rename(
        {
            "Total Value (£)": "Value",
            "Share Price (£)": "Unitary Value",
            "No. Shares": "Units",
            "Pot": "Fund",
            "Description": "Type",
            "Investment": "Narrative",
        },
        axis=1,
    )
    df = df.set_index("Date", drop=True).sort_index()
    df["Asset Code"].fillna("", inplace=True)

    index = ["Date", "Asset Code", "Type", "Narrative"]
    if include_fund:
        index = index + ["Fund"]
    df_piv = pd.DataFrame(
        pd.pivot_table(
            df, index=index, values=["Units", "Value", "Unitary Value"], aggfunc={"Units": sum, "Value": sum, "Unitary Value": np.mean}
        ).to_records()
    ).sort_values("Date")
    df_piv = df_piv.set_index("Date", drop=True)

    to_type = {"Purchase": "BOT", "Sale": "SLD", "Dividend": "DIV", "Fee": "FEE", "Interest": "INT"}
    df_piv.Type = df_piv.Type.map(to_type)
    df_piv.Value = [re_sign_values(x, t) for (x, t) in zip(df_piv.Value, df_piv.Type)]
    df_piv.Units = [re_sign_values(x, t) for (x, t) in zip(df_piv.Units, df_piv.Type)]

    return df_piv


def get_dividends(df, fee_for_avg_holding_period, start, end, SEDOL_MAP, to_sedol={}):
    df_piv_div = df[(df.Type == "DIV") & (df.index <= end) & (df.index > start)]
    if all(df_piv_div.Narrative.str.startswith("Dividend")):
        df_piv_div["Asset Code"] = df_piv_div["Narrative"].str[9:16]
    else:
        df_piv_div["Asset Code"] = df_piv_div["Asset Code"].map(to_sedol)

    df_piv_div["Asset Name"] = [SEDOL_MAP.loc[sedol].FULL_NAME if sedol in SEDOL_MAP.index else "N/A" for sedol in df_piv_div["Asset Code"]]
    df_piv_div = pd.pivot_table(df_piv_div.reset_index(), index=["Date", "Asset Code", "Asset Name"], values="Value", aggfunc=sum)
    df_piv_div = df_piv_div.reset_index().set_index("Date")
    if fee_for_avg_holding_period + df_piv_div.Value.sum() < 0:
        raise Exception("Fees are higher than Dividend, cannot embed the fees in the dividends.")
    fee_ratio = 1 + fee_for_avg_holding_period / df_piv_div.Value.sum()
    df_piv_div.Value *= fee_ratio
    df_piv_div.Value = df_piv_div.Value.apply(lambda x: np.round(x, 2))
    return df_piv_div.rename({"Asset Code": "SEDOL", "Value": "Dividend"}, axis=1)


def get_tax_tbl(df_piv_no_div):
    tax_tables = []
    for asset_code in df_piv_no_div["Asset Code"].unique():
        if asset_code == "CASH":
            continue
        df_small = df_piv_no_div[(df_piv_no_div["Asset Code"] == asset_code)]
        df_mini = pd.pivot_table(
            pd.DataFrame(df_small.to_records()),
            index="Date",
            values=["Units", "Unitary Value"],
            aggfunc={"Units": sum, "Unitary Value": np.mean},
        )
        df_mini = df_mini[abs(df_mini.Units) > 0.0001]
        try:
            tax_tbl = get_taxable_event_from_single_asset(df_mini)
            tax_tbl["Asset Code"] = asset_code
            tax_tables.append(tax_tbl)
        except Exception as e:
            print(f"Failed to process: {asset_code}: {e}")
    return pd.concat(tax_tables).sort_index()


def get_tax_report(df_tax, start, end, sedol_map, to_sedol=None):
    tax_output = df_tax[(df_tax.index <= end) & (df_tax.index > start)]
    tax_output = pd.DataFrame(tax_output.to_records())
    tax_output.holding_period = tax_output.holding_period.round("1D")
    tax_output["sale_date"] = tax_output.purchase_date + tax_output.holding_period
    tax_output = tax_output.astype(
        {"purchase_price": "float", "sale_price": "float", "taxable_amount": "float", "units_sold": "float"}
    ).round(2)
    tax_output = tax_output[["Asset Code", "purchase_date", "sale_date", "units_sold", "purchase_price", "sale_price", "taxable_amount"]]
    if to_sedol is not None:
        tax_output["Asset Code"] = tax_output["Asset Code"].map(to_sedol)
    tax_output["Asset Name"] = [sedol_map.loc[sedol].FULL_NAME if sedol in sedol_map.index else "N/A" for sedol in tax_output["Asset Code"]]
    tax_output = tax_output.rename(
        columns={
            "Asset Code": "SEDOL",
            "units_sold": "Units",
            "purchase_date": "Purchase Date",
            "sale_date": "Sale Date",
            "purchase_price": "Purchase Price",
            "sale_price": "Sale Price",
            "taxable_amount": "Taxable Amount",
        }
    )
    return tax_output[["SEDOL", "Asset Name", "Purchase Date", "Sale Date", "Units", "Purchase Price", "Sale Price", "Taxable Amount"]]


def get_report_small(tax_tbl, start, end, sedol_map, to_sedol=None):
    tax = tax_tbl[(tax_tbl.index <= end) & (tax_tbl.index > start)]
    tax = pd.DataFrame(tax.to_records())

    tax_small = pd.pivot_table(
        tax,
        index="Asset Code",
        values=["units_sold", "sale_price", "purchase_price", "purchase_date", "taxable_amount", "holding_period"],
        aggfunc={"units_sold": sum, "sale_price": np.mean, "purchase_price": np.mean, "taxable_amount": sum, "holding_period": np.mean},
    )
    tax_small = pd.DataFrame(tax_small.to_records())

    if to_sedol is not None:
        tax_small["Asset Code"] = tax_small["Asset Code"].map(to_sedol)
    tax_small["Asset Name"] = [sedol_map.loc[sedol].FULL_NAME if sedol in sedol_map.index else "N/A" for sedol in tax_small["Asset Code"]]
    tax_small.holding_period = tax_small.holding_period.round("1D")
    tax_small = tax_small.astype({"taxable_amount": "float", "units_sold": "float"}).round(2)
    tax_small = tax_small.rename(
        columns={
            "Asset Code": "SEDOL",
            "units_sold": "Units",
            "holding_period": "Average Holding Period",
            "purchase_price": "Purchase Price",
            "sale_price": "Sale Price",
            "taxable_amount": "Taxable Amount",
        }
    )
    return tax_small[["SEDOL", "Asset Name", "Units", "Average Holding Period", "Purchase Price", "Sale Price", "Taxable Amount"]]


def get_relevant_purchases_for_sale(sale_units: float, purchases: dict) -> dict:
    if sale_units == 0:
        return []

    relevant_purchases = {}
    remaining_sale_units = sale_units
    for purchase_date, purchase_unit in purchases.items():
        if remaining_sale_units > purchase_unit:
            relevant_purchases[purchase_date] = purchase_unit
            remaining_sale_units -= purchase_unit
        else:
            relevant_purchases[purchase_date] = remaining_sale_units
            break

    relevant_sum = sum([v for k, v in relevant_purchases.items()])
    assert abs(sale_units - relevant_sum) < __EPSILON, f"{sale_units} vs {relevant_sum}"
    return relevant_purchases


def get_remaining_purchases(relevant_purchases: dict, purchases: dict) -> dict:
    if len(relevant_purchases) == 0:
        raise Exception(f"relevant_purchases is empty")

    remaining_purchases = {}
    for purchase_date, purchase_unit in purchases.items():
        if purchase_date not in relevant_purchases:
            remaining_purchases[purchase_date] = purchase_unit
        else:
            remaining_purchases[purchase_date] = purchase_unit - relevant_purchases[purchase_date]

    purchase_sum = sum([v for k, v in purchases.items()])
    relevant_remaining_sum = sum([v for k, v in relevant_purchases.items()]) + sum([v for k, v in remaining_purchases.items()])
    assert abs(purchase_sum - relevant_remaining_sum) < __EPSILON, f"{purchase_sum} vs {relevant_remaining_sum}"
    return remaining_purchases


def get_taxable_event_from_single_asset(df: pd.DataFrame) -> pd.DataFrame:
    assert set(df.columns) == set(["Units", "Unitary Value"])

    transactions_history = {}
    for date in df.index:
        transactions_history[date] = df[df.index <= date].to_dict()

    for date, transactions in transactions_history.items():
        purchases = {}
        sales = {}
        for transaction_date, transaction_unit in transactions["Units"].items():
            if transaction_unit > 0:
                purchases[transaction_date] = transaction_unit
            else:
                sales[transaction_date] = -transaction_unit

        transactions["Purchases"] = purchases
        transactions["Sales"] = sales

        remaining_purchases = purchases
        for sale_unit in list(transactions["Sales"].values())[:-1]:
            remaining_purchases = get_remaining_purchases(
                get_relevant_purchases_for_sale(sale_unit, remaining_purchases), remaining_purchases
            )

        if len(transactions["Sales"].values()) > 0:
            relevant_purchase_for_last_sale = get_relevant_purchases_for_sale(list(transactions["Sales"].values())[-1], remaining_purchases)
            remaining_purchases = get_remaining_purchases(relevant_purchase_for_last_sale, remaining_purchases)
            transactions["Relevant_Purchases"] = relevant_purchase_for_last_sale
        else:
            transactions["Relevant_Purchases"] = {}

        transactions["Remaining_Purchases"] = remaining_purchases

    sales = {}

    for transaction_date, transaction in transactions_history.items():
        if transaction_date not in transaction["Sales"].keys():
            continue  # this is a purchase, not taxable event

        units_sold = transaction["Sales"][transaction_date]
        sale_price = transaction["Unitary Value"][transaction_date]
        relevant_purchase = transaction["Relevant_Purchases"]
        relevant_purchase = [(d, u, transaction["Unitary Value"][d]) for (d, u) in relevant_purchase.items()]

        assert abs(units_sold - sum([u for (d, u, v) in relevant_purchase])) < __EPSILON

        purchase_price = sum([u * v for (d, u, v) in relevant_purchase]) / units_sold
        purchase_date = [d for d, u, v in relevant_purchase if u != 0.0][0]

        sales[transaction_date] = {
            "units_sold": units_sold,
            "sale_price": sale_price,
            "purchase_price": purchase_price,
            "purchase_date": purchase_date,
        }

    sales_df = pd.DataFrame.from_dict(sales).transpose()
    sales_df["taxable_amount"] = sales_df["units_sold"] * (sales_df["sale_price"] - sales_df["purchase_price"])
    sales_df["holding_period"] = sales_df.index - sales_df["purchase_date"]

    return sales_df


def get_capital_gain_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.Type != "DIV"]
    df = df[df.Type != "FEE"]
    df = df[df.Type != "INT"]
    tax_tables = []
    for asset_code in df["Asset Code"].unique():
        if (asset_code == "CASH") or (asset_code == ""):
            continue
        df_small = df[(df.Narrative.str.contains(asset_code)) & (df["Asset Code"] != "CASH") & (df["Asset Code"] != "")]
        df_mini = pd.pivot_table(
            pd.DataFrame(df_small.to_records()),
            index="Date",
            values=["Units", "Unitary Value"],
            aggfunc={"Units": sum, "Unitary Value": np.mean},
        )
        df_mini = df_mini[abs(df_mini.Units) > 0.0001]
        try:
            tax_tbl = get_taxable_event_from_single_asset(df_mini)
            tax_tbl["Asset Code"] = asset_code
            tax_tables.append(tax_tbl)
        except Exception as e:
            print(f"Failed to process: {asset_code}: {e}")
    return pd.concat(tax_tables).sort_index()


def __get_event_info(r):
    d1 = r["Purchase Date"]
    d2 = r["Sale Date"]
    v1 = r["Purchase Price"]
    v2 = r["Sale Price"]
    return ([d1, d2], [v1, v2])


def plot_asset_life(df_piv, tax_full_output, asset, start_date, to_sedol):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure

    df_asset = df_piv[(df_piv["Asset Code"] == asset) & (df_piv.Type != "DIV")]
    if asset not in to_sedol:
        raise KeyError(f"{asset} is not in to_sedol.")
    tax_event = tax_full_output[tax_full_output.SEDOL == to_sedol[asset]].reset_index()

    asset_value = df_asset["Unitary Value"]
    asset_units = df_asset["Units"]

    max_size = max(abs(asset_units.values))
    size = [(abs(x) + 1) / (max_size + 1) * 100 for x in asset_units.values]
    color = ["r" if x == "SLD" else "g" for x in df_asset.Type]

    figure(figsize=(15, 6), dpi=80)
    plt.scatter(asset_value.index, asset_value.values, size, color=color)
    plt.plot(asset_value, alpha=0.5)
    plt.axvspan(start_date, max(asset_value.index), facecolor="b", alpha=0.1)
    for _, r in tax_event.iterrows():
        plt.plot(*__get_event_info(r), "--", color="gold", alpha=0.5)
    plt.title(f"{asset} price with BUY and SALE")
    return plt
