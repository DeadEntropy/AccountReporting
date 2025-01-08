import configparser
from datetime import datetime

import pandas as pd
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
            .sort_values(["AssetMapped", "Date"], ascending=[True, False])
            .reset_index(drop=True)
            .dropna()
        )

        df["AssetPriceChangeInRefCurrency"] = df.groupby("AssetMapped")["AssetPriceInRefCurrency"].diff()

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


class DataServer:
    """Class to prepare the data for the UI"""

    __agg_func = {
        "Quantity": ["sum", list],
        "MemoMapped": list,
        "Type": list,
        "SubType": list,
        "AssetPriceInRefCurrency": "first",
        "AssetPriceChangeInRefCurrency": "first",
    }

    def __init__(self, data_manager: DataManager, data_market: DataMarket):
        self.data_manager = data_manager
        self.data_market = data_market

        self._df_grouped_transactions = None
        self._df_transaction_price = None

        self.index_names = ["Account", "Asset", "Date", "MemoMapped"]
        self.sum_names = ["Quantity"]

    def group_transaction(self):
        """group the transaction by ["Account", "AssetMapped", "Date"]"""
        df_grouped_transactions = self.data_manager.transactions.groupby(["Account", "Asset", "Date"]).agg(
            {k: DataServer.__agg_func[k] for k in DataServer.__agg_func if k in self.data_manager.transactions.columns}
        )
        df_grouped_transactions.columns = [
            "_".join(col).strip() if isinstance(col, tuple) else col for col in df_grouped_transactions.columns
        ]
        df_grouped_transactions = df_grouped_transactions.reset_index()
        df_grouped_transactions["AssetMapped"] = df_grouped_transactions["Asset"].map(self.data_market.get_asset_map(self.data_manager))
        df_grouped_transactions = df_grouped_transactions.set_index(["Account", "AssetMapped", "Date"]).drop(columns="Asset")
        self._df_grouped_transactions = df_grouped_transactions

    def get_values_by_asset(self, date_range: list = None, account: str | list = None):
        """Retrieve asset values and related financial metrics within a specified date range and/or for a specific account."""

        if account is not None:
            if isinstance(account, str):
                account = [account]
            df = self._df_grouped_transactions.loc[account].groupby(["AssetMapped", "Date"]).agg("sum")
            prices = self.data_market.prices.loc[df.reset_index().AssetMapped.unique()].reset_index()
            prices = prices[prices.Date > df.reset_index().Date.min()].set_index(["AssetMapped", "Date"])
        else:
            df = self._df_grouped_transactions
            df = df.groupby(["AssetMapped", "Date"]).agg("sum")
            prices = self.data_market.prices

        df_prices = pd.merge(df, prices, left_index=True, right_index=True, how="outer").fillna({"Quantity_sum": 0})
        df_prices["Quantity_cumsum"] = df_prices.groupby(["AssetMapped"])["Quantity_sum"].cumsum()
        df_prices["Value"] = df_prices["Quantity_cumsum"] * df_prices["AssetPriceInRefCurrency"]
        df_prices["CapitalGain"] = df_prices["Quantity_cumsum"] * df_prices["AssetPriceChangeInRefCurrency"]

        df_prices["Quantity_list"] = df_prices["Quantity_list"].apply(lambda d: d if isinstance(d, list) else [])
        df_prices["MemoMapped_list"] = df_prices["MemoMapped_list"].apply(lambda d: d if isinstance(d, list) else [])
        df_prices["Type_list"] = df_prices["Type_list"].apply(lambda d: d if isinstance(d, list) else [])
        df_prices["SubType_list"] = df_prices["SubType_list"].apply(lambda d: d if isinstance(d, list) else [])

        if date_range is not None:
            if len(date_range) != 2:
                raise ValueError("date_range must be a list of two dates")
            return df_prices[(df_prices.Date >= date_range[0]) & (df_prices.Date <= date_range[1])]
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
        return DataServer.consolidate_transactions([(k, d[k][0], d[k][1], d[k][2]) for d in l for k in d])

    def get_values_timeseries(self, date_range: list = None, account: str | list = None) -> pd.DataFrame:
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
            try:
                return "<br>".join([f"{v:>7,.0f}: {k[:max_char] if isinstance(k,str) else ''}" for (k, v, t, s) in labels]) + total_label
            except TypeError:
                return ""

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

    def get_figure_timeseries(self, date_range: list = None, account: str | list = None, annotation_count: int = 3) -> go.Figure:
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
