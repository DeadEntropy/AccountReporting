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

    return date_column.map(convert_date)


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

    @property
    def assets(self) -> dict:
        """returns a series of asset and their first transaction date"""
        return self.transactions.groupby("Asset").Date.min().to_dict()


class DataMarket:
    """Class to handle the interaction of the data with the market prices"""

    def __init__(self, data_manager: DataManager, ref_currency: str, config=None):
        self.data_manager = data_manager
        self.config = config
        self.ref_currency = ref_currency
        self.prices = None

    def get_asset_map(self) -> dict:
        """returns a map of the assets to their market symbol"""
        loader = market_loader.MarketLoader()
        return {
            instr: (loader.get_symbol(instr, "USD") if instr not in loader.source_map.keys() else instr)
            for instr in self.data_manager.transactions.Asset.dropna().unique()
        }

    def load_prices(self) -> None:
        """loads market prices from sources specified in the config file"""

        loader = market_loader.MarketLoader(self.config)
        assets = self.data_manager.assets
        period = {a: "10y" if (datetime.today().date() - assets[a]).days / 356 < 10 else "max" for a in assets}
        market_prices = loader.load(assets, self.ref_currency, period)

        df_prices = DataMarket.dataframe_from_nested_dict(market_prices)
        df_prices["Date"] = normalize_date_column(df_prices["Date"])
        df_prices = df_prices[df_prices.Price != 0]

        df_fx = pd.concat(
            [df_prices[df_prices.AssetMapped.str.startswith(ccy)] for ccy in df_prices.Currency.unique() if ccy != self.ref_currency]
        )
        df_fx["Asset"] = df_fx["AssetMapped"].str.replace(f"{self.ref_currency}=X", "")  # TODO properly parse the asset to extract the ccy
        df_fx.rename(
            {"Currency": "RefCurrency", "Asset": "Currency", "Price": "Fx"},
            axis=1,
            inplace=True,
        )

        df_dummy = pd.DataFrame({"Date": pd.date_range(assets[self.ref_currency], datetime.today().date())})
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

        asset_map = self.data_market.get_asset_map()

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
            aggfunc={"Quantity": ["sum", list], "MemoMapped": list, "Type": list},
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

        return merged

    @staticmethod
    def get_full_range(df1, df2):
        """return the full date range covering df1.Date and df2.Date, up to today"""
        return pd.date_range(df1.Date.min(), max(datetime.today().date(), df2.Date.max()), name="Date")

    @staticmethod
    def consolidate_label(label):
        """Aggregate duplicate labels and sort by absolute value"""
        d = {}
        for k, v, c in label:
            if k not in d:
                d[k] = v, c
            else:
                d[k] = (d[k][0] + v, c)
        return sorted([(k, v[0], v[1]) for k, v in d.items()], key=lambda item: -abs(item[1]))

    @staticmethod
    def aggregate_label(l):
        """Aggregate a list of labels into a single label"""
        return DataServer.consolidate_label([(k, d[k][0], d[k][1]) for d in l for k in d])

    def get_values_timeseries(self, date_range: list = None, account: str = None) -> pd.DataFrame:
        """returns a timeseries of the values"""
        df_asset = self.get_values_by_asset(date_range, account)

        df_asset["TransactionValue_list"] = [
            {m: (q * p, t) for m, q, t in zip(m_list, q_list, t_list)}
            for m_list, q_list, p, t_list in zip(
                df_asset["MemoMapped_list"], df_asset["Quantity_list"], df_asset["AssetPriceInRefCurrency"], df_asset["Type_list"]
            )
        ]
        df_values_timeseries = (
            df_asset.groupby("Date")
            .agg({"Value": "sum", "TransactionValue_list": DataServer.aggregate_label})
            .rename(columns={"TransactionValue_list": "Label"})
        )

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

    def prepare_timeseries_label(self, labels, max_labels: int = 20, max_char: int = 20):
        """prepares the label for the hover"""
        total_amt = sum([l[1] for l in labels])
        total_label = f"<br><br>TOTAL: {total_amt:,.0f}" if total_amt != 0 else ""

        if len(labels) < max_labels:
            return "<br>".join([f"{v:>7,.0f}: {k[:max_char]}" for (k, v, t) in labels]) + total_label

        threshold = sorted([abs(v) for (k, v, t) in labels], reverse=True)[max_labels - 1]
        largest_labels = [(k, v) for (k, v, t) in labels if abs(v) > threshold]
        remainging_amount = sum([v for k, v, t in labels]) - sum([v for k, v in largest_labels])
        largest_labels = sorted(
            {**dict(largest_labels), **{"OTHER": remainging_amount}}.items(),
            key=lambda item: -abs(item[1]),
        )
        return "<br>".join([f"{v:>7,.0f}: {k[:25]}" for (k, v) in largest_labels]) + total_label

    def get_timeseries_annotations(self, labels: pd.Series, annotation_count: int = 3) -> list:
        """returns the annotations for the figure"""
        result = [(index, *tup) for index, sublist in labels.items() for tup in sublist if tup[2] not in self._iat_types]
        df_annotations = (
            pd.DataFrame(result, columns=["Date", "MemoMapped", "TransactionValue", "Type"])
            .sort_values(by="TransactionValue", ascending=True)
            .iloc[:annotation_count]
            .reset_index(drop=True)
        )

        df_annotations["Annotation"] = [
            f"{memo[:20]}: {tv:,.0f}" for memo, tv in zip(df_annotations["MemoMapped"], df_annotations["TransactionValue"])
        ]

        return df_annotations.groupby("Date").agg({"Annotation": lambda x: "<br>".join(x)})

    def get_figure_timeseries(self, date_range: list = None, account: str = None, annotation_count: int = 3) -> pd.DataFrame:
        """plots the timeseries of the account value"""
        df_values_timeseries = self.data_server.get_values_timeseries(date_range, account)
        df_annotations = self.get_timeseries_annotations(df_values_timeseries["Label"], annotation_count=annotation_count)

        fig = go.Figure(
            data=go.Scatter(
                x=df_values_timeseries.index,
                y=df_values_timeseries["Value"],
                hovertext=df_values_timeseries["Label"].apply(self.prepare_timeseries_label),
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
