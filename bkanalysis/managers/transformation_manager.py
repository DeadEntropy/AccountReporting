from datetime import datetime

import pandas as pd

from bkanalysis.managers.data_manager import DataManager
from bkanalysis.managers.market_manager import MarketManager


class TransformationManager:
    """Class to prepare the data for the UI"""

    __agg_func = {
        "Quantity": ["sum", list],
        "MemoMapped": list,
        "Type": list,
        "SubType": list,
        "AssetPriceInRefCurrency": "first",
        "AssetPriceChangeInRefCurrency": "first",
    }

    def __init__(self, data_manager: DataManager, market_manager: MarketManager):
        self.data_manager = data_manager
        self.market_manager = market_manager

        self._df_grouped_transactions = None

        # result cache for the expensive transformations; only valid until the
        # underlying data changes, i.e. cleared whenever group_transaction() re-runs.
        # Not locked: concurrent computes of the same key produce identical values.
        self._cache = {}

        self.index_names = ["Account", "Asset", "Date", "MemoMapped"]
        self.sum_names = ["Quantity"]

        self.group_transaction()

    @staticmethod
    def __key_part(value):
        """normalize an argument into a hashable cache-key part"""
        if isinstance(value, (list, tuple)):
            return tuple(TransformationManager.__key_part(v) for v in value)
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.isoformat()
        return value

    def group_transaction(self):
        """group the transaction by ["Account", "AssetMapped", "Date"]"""
        self._cache.clear()
        df_grouped_transactions = self.data_manager.transactions.groupby(["Account", "Asset", "Date"]).agg(
            {
                k: TransformationManager.__agg_func[k]
                for k in TransformationManager.__agg_func
                if k in self.data_manager.transactions.columns
            }
        )
        df_grouped_transactions.columns = [
            "_".join(col).strip() if isinstance(col, tuple) else col for col in df_grouped_transactions.columns
        ]
        df_grouped_transactions = df_grouped_transactions.reset_index()
        df_grouped_transactions["AssetMapped"] = df_grouped_transactions["Asset"].map(self.market_manager.get_asset_map(self.data_manager))
        df_grouped_transactions = df_grouped_transactions.set_index(["Account", "AssetMapped", "Date"]).drop(columns="Asset")
        self._df_grouped_transactions = df_grouped_transactions

    def get_all_categories(self, date_range, threshold=1000):
        """Returns the complete list of categories based on FullMasterType, FullType and FullSubType"""
        df = self.data_manager.transactions[
            (self.data_manager.transactions.Date >= date_range[0]) & (self.data_manager.transactions.Date <= date_range[1])
        ]
        df = df[(df.FullMasterType != "Intra-Account Transfers")]

        df_fmt = pd.pivot_table(
            df,
            index="FullMasterType",
            values=["Quantity", "FullType"],
            aggfunc={"Quantity": "sum", "FullType": lambda x: len(x.unique())},
        ).sort_values("Quantity")
        df_fmt["Quantity"] = df_fmt["Quantity"].apply(abs)
        df_fmt = df_fmt[(df_fmt.Quantity > threshold) & (df_fmt.FullType > 1)].sort_values("Quantity", ascending=False)
        df_fmt = pd.DataFrame(df_fmt.to_records()).rename({"FullMasterType": "index", "Quantity": "size", "FullType": "count"}, axis=1)
        df_fmt["index"] = [f"MasterType: {k}" for k in df_fmt["index"]]

        df_ft = pd.pivot_table(
            df,
            index="FullType",
            values=["Quantity", "FullSubType"],
            aggfunc={"Quantity": "sum", "FullSubType": lambda x: len(x.unique())},
        ).sort_values("Quantity")
        df_ft["Quantity"] = df_ft["Quantity"].apply(abs)
        df_ft = df_ft[(df_ft.Quantity > threshold) & (df_ft.FullSubType > 1)].sort_values("Quantity", ascending=False)
        df_ft = pd.DataFrame(df_ft.to_records()).rename({"FullType": "index", "Quantity": "size", "FullSubType": "count"}, axis=1)
        df_ft["index"] = [f"Type: {k}" for k in df_ft["index"]]

        df_st = pd.pivot_table(
            df,
            index="FullSubType",
            values=["Quantity", "MemoMapped"],
            aggfunc={"Quantity": "sum", "MemoMapped": lambda x: len(x.unique())},
        ).sort_values("Quantity")
        df_st["Quantity"] = df_st["Quantity"].apply(abs)
        df_st = df_st[(df_st.Quantity > threshold) & (df_st.MemoMapped > 1)].sort_values("Quantity", ascending=False)
        df_st = pd.DataFrame(df_st.to_records()).rename({"FullSubType": "index", "Quantity": "size", "MemoMapped": "count"}, axis=1)
        df_st["index"] = [f"SubType: {k}" for k in df_st["index"]]

        return pd.concat([df_ft, df_st])["index"]

    def get_values_by_asset(self, date_range: list = None, account: str | list = None):
        """Retrieve asset values and related financial metrics within a specified date range and/or for a specific account.

        Results are cached until group_transaction() re-runs; a copy is returned so callers can mutate it freely."""
        key = ("values_by_asset", TransformationManager.__key_part(date_range), TransformationManager.__key_part(account))
        if key not in self._cache:
            self._cache[key] = self.__compute_values_by_asset(date_range, account)
        return self._cache[key].copy()

    def __compute_values_by_asset(self, date_range: list, account: str | list):
        df_prices = self.__get_values_by_asset_full(account)

        if date_range is not None:
            if len(date_range) != 2:
                raise ValueError("date_range must be a list of two dates")
            df_prices = df_prices.reset_index()
            df_prices = df_prices[(df_prices.Date >= date_range[0]) & (df_prices.Date <= date_range[1])]
            return df_prices.set_index(["AssetMapped", "Date"])
        return df_prices

    def __get_values_by_asset_full(self, account: str | list):
        """full-history asset values for the given account filter; cached separately because the
        merge + cumsum is the expensive step and any date_range is a cheap slice of this frame"""
        key = ("values_by_asset_full", TransformationManager.__key_part(account))
        if key in self._cache:
            return self._cache[key]

        assert (
            self._df_grouped_transactions is not None
        ), "Transactions were not grouped. have you called .group_transaction() before calling .get_values_by_asset()?"

        if account is not None:
            if isinstance(account, str):
                account = [account]
            df = self._df_grouped_transactions.loc[account].groupby(["AssetMapped", "Date"]).agg("sum")
            prices = self.market_manager.prices.loc[df.reset_index().AssetMapped.unique()].reset_index()
            prices = prices[prices.Date > df.reset_index().Date.min()].set_index(["AssetMapped", "Date"])
        else:
            df = self._df_grouped_transactions
            df = df.groupby(["AssetMapped", "Date"]).agg("sum")
            prices = self.market_manager.prices

        df_prices = pd.merge(df, prices, left_index=True, right_index=True, how="outer").fillna({"Quantity_sum": 0})
        df_prices["Quantity_cumsum"] = df_prices.groupby(["AssetMapped"])["Quantity_sum"].cumsum()
        df_prices["Value"] = df_prices["Quantity_cumsum"] * df_prices["AssetPriceInRefCurrency"]
        df_prices["CapitalGain"] = df_prices["Quantity_cumsum"] * df_prices["AssetPriceChangeInRefCurrency"]

        for col in ["Quantity_list", "MemoMapped_list", "Type_list", "SubType_list"]:
            if col in df_prices.columns:
                df_prices[col] = df_prices[col].apply(lambda d: d if isinstance(d, list) else [])
            else:
                df_prices[col] = [[] for _ in range(len(df_prices))]

        self._cache[key] = df_prices
        return df_prices

    @staticmethod
    def consolidate_transactions(transactions):
        """Aggregate duplicate transactions and sort by absolute value"""
        d = {}
        for k, v, t, s in transactions:
            if k not in d:
                d[k] = v, t, s
            else:
                d[k] = (d[k][0] + v, t, s)
        return sorted([(k, v[0], v[1], v[2]) for k, v in d.items()], key=lambda item: -abs(item[1]))

    @staticmethod
    def aggregate_transactions(l):
        """Aggregate a list of transactions"""
        return TransformationManager.consolidate_transactions([(k, d[k][0], d[k][1], d[k][2]) for d in l for k in d])

    @staticmethod
    def to_agg_dict(m_list, q_list, p, t_list, s_list):
        output = {}
        for m, q, t, s in zip(m_list, q_list, t_list, s_list):
            if m not in output:
                output[m] = (q * p, t, s)
            else:
                output[m] = (output[m][0]  + q * p, t, s)
                
        return output

    def get_values_timeseries(self, date_range: list = None, account: str | list = None) -> pd.DataFrame:
        """returns a timeseries of the values

        Results are cached until group_transaction() re-runs; a copy is returned so callers can mutate it freely."""
        key = ("values_timeseries", TransformationManager.__key_part(date_range), TransformationManager.__key_part(account))
        if key not in self._cache:
            self._cache[key] = self.__compute_values_timeseries(date_range, account)
        return self._cache[key].copy()

    def __compute_values_timeseries(self, date_range: list, account: str | list) -> pd.DataFrame:
        df_asset = self.get_values_by_asset(date_range, account)

        df_asset["TransactionValue_list"] = [
            TransformationManager.to_agg_dict(m_list, q_list, p, t_list, s_list)
            for m_list, q_list, p, t_list, s_list in zip(
                df_asset["MemoMapped_list"],
                df_asset["Quantity_list"],
                df_asset["AssetPriceInRefCurrency"],
                df_asset["Type_list"],
                df_asset["SubType_list"],
            )
        ]
        df_values_timeseries = df_asset.groupby("Date").agg(
            {"Value": "sum", "TransactionValue_list": TransformationManager.aggregate_transactions}
        )

        return df_values_timeseries

    def get_flow_values(
        self,
        date_start=None,
        date_end=None,
        account: str = None,
        how: str = "both",
        include_iat: bool = False,
        include_full_types: bool = True,
    ):
        """returns the flow values (one row per transaction) for the given filters

        Results are cached until group_transaction() re-runs; a copy is returned so callers can mutate it freely."""
        key = (
            "flow_values",
            TransformationManager.__key_part(date_start),
            TransformationManager.__key_part(date_end),
            TransformationManager.__key_part(account),
            how.lower(),
            include_iat,
            include_full_types,
        )
        if key not in self._cache:
            self._cache[key] = self.__compute_flow_values(date_start, date_end, account, how, include_iat, include_full_types)
        return self._cache[key].copy()

    def __compute_flow_values(
        self,
        date_start,
        date_end,
        account: str,
        how: str,
        include_iat: bool,
        include_full_types: bool,
    ):
        if date_end is None and date_start is None:
            date_range = None
        elif date_start is None or date_end is None:
            raise ValueError("date_start and date_end must both have values or both be None.")
        else:
            date_range = [date_start, date_end]

        df_values_timeseries = self.get_values_by_asset(date_range=date_range, account=account)

        df_values_timeseries["Value_list"] = [
            [q * p for q in q_list]
            for p, q_list in zip(df_values_timeseries["AssetPriceInRefCurrency"], df_values_timeseries["Quantity_list"])
        ]
        columns_list = ["MemoMapped_list", "Type_list", "SubType_list", "Value_list"]
        df_values_exploded = df_values_timeseries.explode(columns_list)[columns_list].dropna(subset="Value_list").copy()
        df_values_exploded.columns = [c.replace("_list", "") for c in df_values_exploded.columns]
        df_values_exploded["SubType"] = df_values_exploded["SubType"].fillna(" ")
        df_values_exploded["MemoMapped"] = df_values_exploded["MemoMapped"].fillna(" ")
        df_values_exploded["Type"] = df_values_exploded["Type"].fillna(" ")

        if not include_iat:
            df_flow_values = df_values_exploded[df_values_exploded["Type"] != "IAT"].copy()
        else:
            df_flow_values = df_values_exploded

        if include_full_types:
            df_flow_values["FullType"] = df_flow_values.Type.map(self.data_manager.map_type)
            df_flow_values["FullSubType"] = df_flow_values.SubType.map(self.data_manager.map_subtype)

        if how.lower() == "both":
            return df_flow_values
        elif how.lower() == "in":
            return df_flow_values[df_flow_values["Value"] > 0]
        elif how.lower() == "out":
            return df_flow_values[df_flow_values["Value"] < 0]

        raise ValueError(f"Invalid 'how=' must be 'both', 'in', or 'out' but was {how}.")

    def get_iat_imbalance(self, date_start=None, date_end=None, account: str = None, threshold: float = 1.0):
        """returns the net of the intra-account-transfer legs over the period, plus the per-memo breakdown.

        An intra-account transfer moves value between two tracked accounts, so its legs must sum to
        zero. A non-zero net means one leg is missing or mis-paired: either a counterparty account is
        absent from the data, a transaction is tagged IAT when it is really a flow, or the two legs
        straddle the reporting cut-off. Because IAT legs are excluded from every flow-based figure
        while still moving the account balances, that residual is exactly the amount by which the
        flow-based view fails to reconcile to the change in wealth.

        Returns (net, df_by_memo) where df_by_memo is sorted by absolute imbalance and only keeps
        memos whose net exceeds `threshold`."""
        df = self.get_flow_values(date_start, date_end, account, how="both", include_iat=True)
        df_iat = df[df["Type"] == "IAT"]
        if len(df_iat) == 0:
            return 0.0, pd.DataFrame(columns=["MemoMapped", "Value", "Legs"])

        value = df_iat["Value"].astype(float)
        by_memo = (
            pd.DataFrame({"MemoMapped": df_iat["MemoMapped"].values, "Value": value.values})
            .groupby("MemoMapped")
            .agg(Value=("Value", "sum"), Legs=("Value", "size"))
            .reset_index()
        )
        by_memo = by_memo[by_memo["Value"].abs() > threshold]
        by_memo = by_memo.reindex(by_memo["Value"].abs().sort_values(ascending=False).index).reset_index(drop=True)

        return value.sum(), by_memo

    def __get_price_on_date(self, date: str, threshold: float = 100) -> pd.DataFrame:
        date = pd.Timestamp(date)
        q_t = self._df_grouped_transactions.reset_index()
        q_t = q_t[q_t.Date <= date]
        q_t = q_t.groupby(["Account", "AssetMapped"]).agg({"Quantity_sum": "sum"})

        max_date = self.market_manager.prices.index.levels[1].max()
        if date > max_date:
            date = max_date

        prices = self.market_manager.prices.xs(date, level="Date")

        v_t = pd.merge(q_t.reset_index(), prices.reset_index(), on="AssetMapped", how="left")
        v_t["Price"] = v_t["Quantity_sum"] * v_t["AssetPriceInRefCurrency"]

        df_account = pd.pivot_table(v_t, index="Account", values="Price", aggfunc="sum").sort_values("Price", ascending=False)

        return df_account[df_account.Price.apply(abs) > threshold]

    def get_price_comparison_on_dates(self, date1: str, date2: str, by_account_type=True) -> pd.DataFrame:
        """returns a DataFrame comparing the account type on 2 dates"""
        df1 = self.__get_price_on_date(date1)
        df2 = self.__get_price_on_date(date2)
        df = pd.concat([df1, df2], axis=1).fillna(0.0)
        df.columns = [date1, date2]

        if by_account_type:
            df["AccountType"] = [self.data_manager.map_account_type.loc[x] for x in df.index]
            df = (
                pd.pivot_table(df.reset_index(), index="AccountType", values=[date1, date2], aggfunc="sum")
                .sort_values(date1, ascending=False)
                .reset_index()
            )
            df.columns = ["AccountType"] + [f"{d.date():%b-%y}" for d in df.columns[1:]]
        else:
            df.columns = ["Account"] + [f"{d.date():%b-%y}" for d in df.columns[1:]]

        return df
