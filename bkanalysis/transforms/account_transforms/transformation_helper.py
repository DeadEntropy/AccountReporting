import pandas as pd
import datetime as dt
from bkanalysis.market.market import Market


def get_key(proportion, date):
    return max([v for v in list(proportion.keys()) if date > pd.to_datetime(v)])


def __to_float(s):
    try:
        return s.str.replace("£", "").str.replace(",", "").astype("float64")
    except AttributeError:
        return s


SWITCH_DATE_FORMAT = "%Y-%m-%d"


def get_transaction(df: str, market: Market, proportion: {}, switch: {}, ref_currency: str, memo: str, date_format: str = "%d/%m/%Y"):
    df_transaction = df.fillna(0.0)

    df_transaction["Date"] = pd.to_datetime(df_transaction["Date"], format=date_format)
    df_transaction["Amount"] = __to_float(df_transaction["Amount"]).fillna(0.0)
    df_transaction = df_transaction.set_index("Date")

    list_of_funds = list(set([item for sublist in proportion.values() for item in sublist]))
    list_of_dfs = [df_transaction]

    for fund_name in list_of_funds:
        df_fund = pd.DataFrame(columns=df_transaction.columns, index=df_transaction.index)
        df_fund["Price"] = [market.get_price_in_currency(fund_name, date, ref_currency) for date in df_transaction.index]
        df_fund["Amount"] = (
            df_transaction[f"Amount"]
            / df_fund["Price"]
            * [proportion[get_key(proportion, date)][fund_name] for date in df_transaction.index]
        )
        df_fund.drop("Price", axis=1, inplace=True)
        df_fund["Account"] = df_transaction["Account"]
        df_fund["AccountType"] = df_transaction["AccountType"]
        df_fund["Currency"] = fund_name
        df_fund["Subcategory"] = f"{memo} Fund Purchase"
        df_fund["Memo"] = f"{memo} Fund Purchase"

        df_cash_impact = pd.DataFrame(columns=df_transaction.columns, index=df_transaction.index)

        df_cash_impact["Amount"] = -df_transaction[f"Amount"] * [
            proportion[get_key(proportion, date)][fund_name] for date in df_transaction.index
        ]
        df_cash_impact["Account"] = df_transaction["Account"]
        df_cash_impact["AccountType"] = df_transaction["AccountType"]
        df_cash_impact["Currency"] = ref_currency
        df_cash_impact["Subcategory"] = f"{memo} Fund Purchase"
        df_cash_impact["Memo"] = f"{memo} Fund Purchase"

        list_of_dfs.append(df_fund)
        list_of_dfs.append(df_cash_impact)

    result = pd.concat(list_of_dfs)
    result = result[result.Amount != 0].reset_index()

    for k, v in switch.items():
        for fund_name, fund_units in v.items():
            row = [
                dt.datetime.strptime(k, SWITCH_DATE_FORMAT),
                df_transaction["Account"].iloc[0],
                fund_units,
                f"{memo} Fund Switch",
                f"{memo} Fund Switch",
                fund_name,
                df_transaction["AccountType"].iloc[0],
            ]
            result = pd.concat([result, pd.DataFrame([row], columns=result.columns)])

    result["Date"] = pd.to_datetime(result["Date"], format=date_format)
    result = result.sort_values("Date", ascending=False).reset_index(drop=True)

    return result
