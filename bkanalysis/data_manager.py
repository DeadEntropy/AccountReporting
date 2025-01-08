import configparser
from datetime import datetime

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from bkanalysis.config import config_helper
from bkanalysis.market import market_loader
from bkanalysis.process import process
from bkanalysis.transforms import master_transform
from bkanalysis.process.iat_identification import IatIdentification

USE_GO = True


def normalize_date_column(date_column):
    """Normalizes the date column to a consistent date format."""

    def convert_date(value):
        if isinstance(value, pd.Timestamp) or isinstance(value, datetime):
            return value.date()
        try:
            return pd.to_datetime(value).date()
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid date format: {value}") from exc

    return pd.to_datetime(date_column.map(convert_date))


class DataManager:
    """Main Data Handling Class."""

    COLUMNS = [
        "Date",
        "Account",
        "Amount",
        "Subcategory",
        "Memo",
        "Currency",
        "MemoSimple",
        "MemoMapped",
        "Type",
        "FullType",
        "SubType",
        "FullSubType",
        "MasterType",
        "FullMasterType",
        "AccountType",
        "FacingAccount",
        "SourceFile",
    ]

    def __init__(self, config=None):
        self._load_config(config)
        self.include_xls = True
        self.include_json = True
        self.ignore_overrides = True
        self.remove_offsetting = True

        self.transactions = None

    @property
    def accounts(self):
        """returns the list of all accounts"""
        return self.transactions.Account.unique()

    @property
    def assets(self) -> dict:
        """returns a series of asset and their first transaction date"""
        return self.transactions.groupby("Asset").Date.min().to_dict()

    def to_disk(self, path: str) -> None:
        """saves the transactions to disk"""
        self.transactions.to_csv(path, index=False)

    def load_pregenerated_data(self, path) -> None:
        """loads the pregenerated data from disk"""
        self.transactions = pd.read_csv(path, parse_dates=["Date"])

    def _load_config(self, config=None):
        if config is None:
            self.config = configparser.ConfigParser()
            if len(self.config.read(config_helper.source)) != 1:
                raise OSError(f"no config found in {config_helper.source}")
        else:
            self.config = config

    def load_data_from_disk(self) -> None:
        """loads the account data from disk. the location needs to be specified in the config file"""
        df_raw = self._load_raw_data_from_disk()
        self.transactions: pd.DataFrame = self._enrich_raw_data(df_raw)[DataManager.COLUMNS].rename(
            {"Currency": "Asset", "Amount": "Quantity"}, axis=1
        )
        self.transactions["Date"] = normalize_date_column(self.transactions["Date"])

    def _load_raw_data_from_disk(self) -> pd.DataFrame:
        """returns the transaction data from file"""
        mt = master_transform.Loader(self.config)
        return mt.load_all(self.include_xls, self.include_json)

    def _enrich_raw_data(self, df_raw) -> pd.DataFrame:
        """return the transaction data enriched with the mapping"""
        pr = process.Process(self.config)
        return pr.process(
            df_raw,
            ignore_overrides=self.ignore_overrides,
            remove_offsetting=self.remove_offsetting,
        )


class DataMarket:
    """Class to handle the interaction of the data with the market prices"""

    def __init__(self, ref_currency: str, config=None):
        self.config = config
        self.ref_currency = ref_currency
        self.prices = None

    def to_disk(self, path: str) -> None:
        """saves the transactions to disk"""
        self.prices.to_csv(path, index=True)

    def load_pregenerated_data(self, path) -> None:
        """loads the pregenerated data from disk"""
        self.prices = pd.read_csv(path, parse_dates=["Date"])
        self.prices = self.prices.set_index(["AssetMapped", "Date"])

    def get_asset_map(self, data_manager: DataManager) -> dict:
        """returns a map of the assets to their market symbol"""
        loader = market_loader.MarketLoader()
        return {
            instr: (loader.get_symbol(instr, "USD") if instr not in loader.source_map.keys() else instr)
            for instr in data_manager.transactions.Asset.dropna().unique()
        }

    def load_prices(self, data_manager: DataManager) -> None:
        """loads market prices from sources specified in the config file"""

        loader = market_loader.MarketLoader(self.config)
        assets = data_manager.assets
        period = {a: "10y" if (datetime.today() - assets[a]).days / 356 < 10 else "max" for a in assets}
        market_prices = loader.load(assets, self.ref_currency, period)

        df_prices = DataMarket.dataframe_from_nested_dict(market_prices)
        df_prices["Date"] = normalize_date_column(df_prices["Date"])
        df_prices = df_prices[df_prices.Price != 0]

        asset_map = self.get_asset_map(data_manager)
        date_range = {
            asset_map[k]: pd.date_range(
                (
                    start
                    if start < df_prices[df_prices.AssetMapped == asset_map[k]].Date.max()
                    else df_prices[df_prices.AssetMapped == asset_map[k]].Date.min()
                ),
                datetime.today(),
                name="Date",
            )
            for k, start in data_manager.assets.items()
        }

        # Function to apply reindexing per group
        def reindex_group(group, group_name):
            if group_name not in date_range:
                return group
            return group.reset_index().set_index("Date").reindex(date_range[group_name]).ffill().bfill().reset_index()

        df_prices = (
            df_prices.groupby("AssetMapped")
            .apply(lambda group: reindex_group(group, group.name))
            .reset_index(drop=True)[["AssetMapped", "Date", "Currency", "Price"]]
        )

        df_prices = df_prices.dropna()

        df_fx = pd.concat(
            [df_prices[df_prices.AssetMapped.str.endswith("=X")] for ccy in df_prices.Currency.unique() if ccy != self.ref_currency]
        )
        df_fx["Asset"] = df_fx["AssetMapped"].str.replace(f"{self.ref_currency}=X", "")  # TODO properly parse the asset to extract the ccy
        df_fx.rename(
            {"Currency": "RefCurrency", "Asset": "Currency", "Price": "Fx"},
            axis=1,
            inplace=True,
        )

        df_dummy = pd.DataFrame({"Date": pd.date_range(assets[self.ref_currency], datetime.today())})
        df_dummy["Date"] = df_dummy["Date"].dt.date
        df_dummy["AssetMapped"] = f"{self.ref_currency}{self.ref_currency}=X"
        df_dummy["RefCurrency"] = self.ref_currency
        df_dummy["Fx"] = 1.0
        df_dummy["Currency"] = self.ref_currency

        df_fx = pd.concat([df_fx, df_dummy])
        df_fx["Date"] = normalize_date_column(df_fx["Date"])

        df = pd.merge(df_prices, df_fx[["Date", "RefCurrency", "Fx", "Currency"]], on=["Date", "Currency"], how="left")
        df["AssetPriceInRefCurrency"] = df.Price * df.Fx

        df = (
            pd.concat(
                [
                    df_fx[df_fx.Date >= min(assets.values())].rename({"Fx": "AssetPriceInRefCurrency"}, axis=1)[
                        ["AssetMapped", "Date", "AssetPriceInRefCurrency"]
                    ],
                    df[["AssetMapped", "Date", "AssetPriceInRefCurrency"]],
                ]
            )
            .drop_duplicates()
            .sort_values(["AssetMapped", "Date"])
            .reset_index(drop=True)
            .dropna()
        )

        self.prices = df.set_index(["AssetMapped", "Date"])

    @staticmethod
    def dataframe_from_nested_dict(data):
        """Create a DataFrame from a dict<asset:dict<date:Price>>"""
        rows = []
        for asset, date_dict in data.items():
            for date, price_obj in date_dict.items():
                row = {
                    "AssetMapped": asset,
                    "Date": date,
                    "Currency": price_obj.currency,
                    "Price": price_obj.value,
                }
                rows.append(row)
        return pd.DataFrame(rows)


class DataServer2:
    """Class to prepare the data for the UI"""

    def __init__(self, data_manager: DataManager, data_market: DataMarket):
        self.data_manager = data_manager
        self.data_market = data_market

        self._reindexed_transactions = None
        self._df_transaction_price = None

        self.index_names = ["Account", "Asset", "Date", "MemoMapped"]
        self.sum_names = ["Quantity"]

    @property
    def accounts(self):
        """returns the list of all accounts"""
        return self.data_manager.accounts

    def reindex_transactions(self) -> None:
        """Reindex the transactions DataFrame to include all dates for each Account and Asset."""
        meta_data_names = [n for n in self.data_manager.transactions.columns if n not in self.sum_names and n not in self.index_names]
        self.data_manager.transactions["Date"] = pd.to_datetime(self.data_manager.transactions["Date"])

        # Step 1: Aggregate duplicates by summing Quantity, keeping other columns as-is
        aggregation = {**{k: "sum" for k in self.sum_names}, **{k: "first" for k in meta_data_names}}
        df = self.data_manager.transactions.groupby(self.index_names, as_index=False).agg(aggregation)

        # Step 2: Initialize an empty list to store reindexed DataFrames
        reindexed_dfs = []

        # Step 3: Process each group (Account, Asset) separately
        for (account, asset), group in df.groupby(["Account", "Asset"]):
            # Determine the full date range for this group
            date_range = pd.date_range(group["Date"].min(), datetime.today(), freq="D")

            # Create a new DataFrame to reindex the group's data
            date_memo_df = pd.DataFrame({"Date": date_range})

            # Merge the full date range with the group to ensure all dates are included
            expanded_group = pd.merge(date_memo_df, group, on="Date", how="left")

            # Add back the Account and Asset columns
            expanded_group["Account"] = account
            expanded_group["Asset"] = asset

            # Append the reindexed group to the list
            reindexed_dfs.append(expanded_group)

        # Step 4: Concatenate all reindexed DataFrames
        df_reindexed = pd.concat(reindexed_dfs)

        # Step 5: Reset the index if needed
        df_reindexed = df_reindexed.reset_index(drop=True)

        self._reindexed_transactions = df_reindexed.set_index(self.index_names[: self.index_names.index("Date") + 1])

    def merge_transaction_price(self) -> None:
        """Merge the reindexed transactions with the market prices."""
        df_temp = self._reindexed_transactions.reset_index()
        df_temp["AssetMapped"] = df_temp["Asset"].map(self.data_market.get_asset_map(self.data_manager))
        df_temp["Date"] = normalize_date_column(df_temp["Date"])
        df_temp = df_temp.set_index(["AssetMapped", "Date"])

        df_prices = self.data_market.prices.reset_index().sort_values(by=["AssetMapped", "Date"], ascending=[True, False])

        # df_prices_asset = df_prices_asset.groupby("AssetMapped").set_index("Date").reindex(date_range).ffill().bfill().reset_index()

        df_prices["Date"] = normalize_date_column(df_prices["Date"])
        df_prices = df_prices.set_index(["AssetMapped", "Date"])

        self._df_transaction_price = pd.merge(df_temp, df_prices, left_index=True, right_index=True, how="left").sort_index()

        self._df_transaction_price["AssetPriceInRefCurrency"] = self._df_transaction_price.groupby("AssetMapped")[
            "AssetPriceInRefCurrency"
        ].ffill()
        self._df_transaction_price["AssetPriceDiffInRefCurrency"] = self._df_transaction_price.groupby("AssetMapped")[
            "AssetPriceInRefCurrency"
        ].diff()

    def get_values_by_asset(self):
        """returns the data_manager.transactions with the market prices"""
        df = pd.pivot_table(
            self._df_transaction_price,
            index=["AssetMapped", "Date"],
            values=["Quantity", "MemoMapped", "Type", "SubType", "AssetPriceInRefCurrency", "AssetPriceDiffInRefCurrency"],
            aggfunc={
                "Quantity": ["sum", list],
                "MemoMapped": list,
                "Type": list,
                "SubType": list,
                "AssetPriceInRefCurrency": "first",
                "AssetPriceDiffInRefCurrency": "first",
            },
        ).sort_index(ascending=False)

        df.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

        df["Quantity_cumsum"] = (
            df.groupby("AssetMapped")["Quantity_sum"].apply(lambda x: x[::-1].cumsum()[::-1]).reset_index(level=0, drop=True)
        )
        df["Value"] = (
            df.groupby("AssetMapped")
            .apply(lambda group: group["Quantity_cumsum"] * group["AssetPriceInRefCurrency_first"])
            .reset_index(level=0, drop=True)
        )
        df["CapitalGain"] = (
            df.groupby("AssetMapped")
            .apply(lambda group: group["Quantity_cumsum"] * group["AssetPriceDiffInRefCurrency_first"])
            .reset_index(level=0, drop=True)
        )

        df["Quantity_list"] = df["Quantity_list"].apply(lambda d: d if isinstance(d, list) else [])
        df["MemoMapped_list"] = df["MemoMapped_list"].apply(lambda d: d if isinstance(d, list) else [])
        df["Type_list"] = df["Type_list"].apply(lambda d: d if isinstance(d, list) else [])
        df["SubType_list"] = df["SubType_list"].apply(lambda d: d if isinstance(d, list) else [])

        df = df.rename(
            columns={
                "AssetPriceDiffInRefCurrency_first": "AssetPriceChangeInRefCurrency",
                "AssetPriceInRefCurrency_first": "AssetPriceInRefCurrency",
            }
        ).reset_index()
        df.Date = pd.to_datetime(df.Date)
        df = df.set_index(["AssetMapped", "Date"]).sort_index(ascending=[True, False])

        return df


class DataServer:
    """Class to prepare the data for the UI"""

    def __init__(self, data_manager: DataManager, data_market: DataMarket):
        self.data_manager = data_manager
        self.data_market = data_market

    @property
    def accounts(self):
        """returns the list of all accounts"""
        return self.data_manager.accounts

    def get_values_by_asset(self, date_range: list = None, account: str = None):
        """returns the data_manager.transactions with the market prices"""
        transaction_prices = {}

        if account is not None:
            if account not in self.accounts:
                raise KeyError(f"account {account} not found.")
            transactions = self.data_manager.transactions[self.data_manager.transactions.Account == account]
        else:
            transactions = self.data_manager.transactions

        asset_map = self.data_market.get_asset_map(self.data_manager)

        # Process each asset individually
        unique_assets = [asset_map[a] for a in transactions["Asset"].unique() if isinstance(a, str)]
        for asset in unique_assets:
            # Filter data for the current asset
            df_trans_asset = transactions[transactions["Asset"].map(asset_map) == asset]
            df_prices_asset = self.data_market.prices.xs(asset, level="AssetMapped")

            # merge the transaction
            transaction_price = self.merge_transaction_asset(df_trans_asset, df_prices_asset.reset_index())
            transaction_price["AssetMapped"] = asset
            transaction_prices[asset] = transaction_price

        # Combine results for all assets
        df_value = pd.concat(transaction_prices.values(), ignore_index=True)

        if date_range is not None:
            if len(date_range) != 2:
                raise ValueError(f"date_range must be a list of two dates")
            return df_value[(df_value.Date >= date_range[0]) & (df_value.Date <= date_range[1])]
        return df_value

    def merge_transaction_asset(self, df_trans_asset, df_prices_asset):
        """merge the transaction and price dataframes, calculate value and capital gain"""
        date_range = DataServer.get_full_range(df_trans_asset, df_prices_asset)

        df_trans_asset_daily = pd.pivot_table(
            df_trans_asset,
            index="Date",
            values=["Quantity", "MemoMapped", "Type", "SubType"],
            aggfunc={"Quantity": ["sum", list], "MemoMapped": list, "Type": list, "SubType": list},
        ).reindex(date_range)
        df_trans_asset_daily.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in df_trans_asset_daily.columns]
        df_trans_asset_daily["Quantity_sum"] = df_trans_asset_daily["Quantity_sum"].fillna(0)
        df_prices_asset = df_prices_asset.set_index("Date").reindex(date_range).ffill().bfill().reset_index()

        merged = pd.merge(
            df_trans_asset_daily.reset_index(), df_prices_asset[["Date", "AssetPriceInRefCurrency"]], on=["Date"], how="left"
        ).sort_values("Date", ascending=False)

        merged["AssetPriceChangeInRefCurrency"] = -merged["AssetPriceInRefCurrency"].diff().fillna(0.0)
        merged["Quantity_cumsum"] = merged["Quantity_sum"][::-1].cumsum()[::-1]
        merged["Value"] = merged["Quantity_cumsum"] * merged["AssetPriceInRefCurrency"]
        merged["CapitalGain"] = merged["Quantity_cumsum"] * merged["AssetPriceChangeInRefCurrency"]

        merged["Quantity_list"] = merged["Quantity_list"].apply(lambda d: d if isinstance(d, list) else [])
        merged["MemoMapped_list"] = merged["MemoMapped_list"].apply(lambda d: d if isinstance(d, list) else [])
        merged["Type_list"] = merged["Type_list"].apply(lambda d: d if isinstance(d, list) else [])
        merged["SubType_list"] = merged["SubType_list"].apply(lambda d: d if isinstance(d, list) else [])

        return merged

    @staticmethod
    def get_full_range(df1, df2):
        """return the full date range covering df1.Date and df2.Date, up to today"""
        return pd.date_range(df1.Date.min(), max(datetime.today(), df2.Date.max()), name="Date")

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
        return DataServer.consolidate_transactions([(k, d[k][0], d[k][1], d[k][2]) for d in l for k in d])

    def get_values_timeseries(self, date_range: list = None, account: str = None) -> pd.DataFrame:
        """returns a timeseries of the values"""
        df_asset = self.get_values_by_asset(date_range, account)

        df_asset["TransactionValue_list"] = [
            {m: (q * p, t, s) for m, q, t, s in zip(m_list, q_list, t_list, s_list)}
            for m_list, q_list, p, t_list, s_list in zip(
                df_asset["MemoMapped_list"],
                df_asset["Quantity_list"],
                df_asset["AssetPriceInRefCurrency"],
                df_asset["Type_list"],
                df_asset["SubType_list"],
            )
        ]
        df_values_timeseries = df_asset.groupby("Date").agg({"Value": "sum", "TransactionValue_list": DataServer.aggregate_transactions})

        return df_values_timeseries

    def get_values_by_types(self, filters) -> pd.DataFrame:
        """returns a breakdown by types"""
        pass

    def get_values_by_month(self, filters) -> pd.DataFrame:
        """returns a breakdown by month"""
        pass


class DataFigure:
    """Class to prepare the data for the UI"""

    def __init__(self, data_server: DataServer):
        self.data_server = data_server
        self._iat_types = IatIdentification.iat_types + ["C", "S"]

    def __prepare_timeseries_label(self, labels, max_labels: int = 20, max_char: int = 20):
        """prepares the label for the hover"""
        total_amt = sum([l[1] for l in labels])
        total_label = f"<br><br>TOTAL: {total_amt:,.0f}" if total_amt != 0 else ""

        if len(labels) < max_labels:
            return "<br>".join([f"{v:>7,.0f}: {k[:max_char]}" for (k, v, t, s) in labels]) + total_label

        threshold = sorted([abs(v) for (k, v, t, s) in labels], reverse=True)[max_labels - 1]
        largest_labels = [(k, v) for (k, v, t, s) in labels if abs(v) > threshold]
        remainging_amount = sum([v for k, v, t, s in labels]) - sum([v for k, v in largest_labels])
        largest_labels = sorted(
            {**dict(largest_labels), **{"OTHER": remainging_amount}}.items(),
            key=lambda item: -abs(item[1]),
        )
        return "<br>".join([f"{v:>7,.0f}: {k[:25]}" for (k, v) in largest_labels]) + total_label

    def __get_timeseries_annotations(self, labels: pd.Series, annotation_count: int = 3) -> list:
        """returns the annotations for the figure"""
        result = [(index, *tup) for index, sublist in labels.items() for tup in sublist if tup[2] not in self._iat_types]
        df_annotations = (
            pd.DataFrame(result, columns=["Date", "MemoMapped", "TransactionValue", "Type", "SubType"])
            .sort_values(by="TransactionValue", ascending=True)
            .iloc[:annotation_count]
            .reset_index(drop=True)
        )

        df_annotations["Annotation"] = [
            f"{memo[:20]}: {tv:,.0f}" for memo, tv in zip(df_annotations["MemoMapped"], df_annotations["TransactionValue"])
        ]

        return df_annotations.groupby("Date").agg({"Annotation": lambda x: "<br>".join(x)})

    def get_figure_timeseries(self, date_range: list = None, account: str = None, annotation_count: int = 3) -> go.Figure:
        """plots the timeseries of the account value"""
        df_values_timeseries = self.data_server.get_values_timeseries(date_range, account)
        df_annotations = self.__get_timeseries_annotations(df_values_timeseries["TransactionValue_list"], annotation_count=annotation_count)

        fig = go.Figure(
            data=go.Scatter(
                x=df_values_timeseries.index,
                y=df_values_timeseries["Value"],
                hovertext=df_values_timeseries["TransactionValue_list"].apply(self.__prepare_timeseries_label),
            ),
        )

        fig.update_layout(
            title="Total Wealth",
            xaxis_title="Date",
            yaxis_title="Currency",
            margin=dict(t=50),
        )

        i = 0
        for annotation_date in df_annotations.index:
            ann_text = df_annotations.loc[annotation_date]["Annotation"]
            fig.add_annotation(
                x=annotation_date,
                y=df_values_timeseries.loc[annotation_date]["Value"],
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

    def get_total_wealth(self, date: datetime = None) -> float:
        """returns the total wealth at the given date"""
        df_values_timeseries = self.data_server.get_values_timeseries()
        if date is None:
            date = datetime.today()
        return df_values_timeseries.loc[date]["Value"]

    def get_figure_sunburst(self, date_range: list = None, account: str = None, annotation_count: int = 3) -> go.Figure:
        pass
