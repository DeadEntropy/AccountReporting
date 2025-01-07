import configparser
from datetime import datetime

import pandas as pd

from bkanalysis.config import config_helper
from bkanalysis.market import market_loader
from bkanalysis.process import process
from bkanalysis.transforms import master_transform


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

    def get_assets(self) -> pd.DataFrame:
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
        assets = self.data_manager.get_assets()
        period = {a: "10y" if (datetime.today().date() - assets[a]).days / 356 < 10 else "max" for a in assets}
        market_prices = loader.load(assets, self.ref_currency, period)

        df_prices = DataMarket.dataframe_from_nested_dict(market_prices)
        df_prices["Date"] = normalize_date_column(df_prices["Date"])

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

        self._df_fx = df_fx
        self.df = df
        self.prices = df

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

    def __init__(self, data_manager: DataManager, data_market: DataMarket, config=None):
        self.data_manager = data_manager
        self.data_market = data_market
        self.config = config

    def get_values_by_asset(self, account: str = None):
        """returns the data_manager.transactions with the market prices"""
        transaction_prices = {}

        asset_map = self.data_market.get_asset_map()

        if account is not None:
            transactions = self.data_manager.transactions[self.data_manager.transactions.Account == account]
        else:
            transactions = self.data_manager.transactions

        # Process each asset individually
        unique_assets = [asset_map[a] for a in transactions["Asset"].unique() if isinstance(a, str)]
        for asset in unique_assets:
            # Filter data for the current asset
            df_trans_asset = transactions[transactions["Asset"].map(asset_map) == asset]
            df_prices_asset = self.data_market.prices[self.data_market.prices["AssetMapped"] == asset]

            # merge the transaction and pro
            transaction_price = self.merge_transaction_asset(df_trans_asset, df_prices_asset)
            transaction_price["AssetMapped"] = asset
            transaction_prices[asset] = transaction_price

        # Combine results for all assets
        return pd.concat(transaction_prices.values(), ignore_index=True)

    def merge_transaction_asset(self, df_trans_asset, df_prices_asset):
        """merge the transaction and price dataframes, calculate value and capital gain"""
        date_range = DataServer.get_full_range(df_trans_asset, df_prices_asset)

        df_trans_asset_daily = pd.pivot_table(
            df_trans_asset, index="Date", values=["Quantity", "MemoMapped"], aggfunc={"Quantity": ["sum", list], "MemoMapped": list}
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

        return merged

    @staticmethod
    def get_full_range(df1, df2):
        """return the full date range covering df1.Date and df2.Date"""
        try:
            return pd.date_range(df1.Date.min(), max(df1.Date.max(), df2.Date.max()), name="Date")
        except TypeError as type_error:
            raise TypeError("") from type_error

    def get_values_timeseries(self, filters) -> pd.DataFrame:
        """returns a timeseries of the values"""
        pass

    def get_values_by_types(self, filters) -> pd.DataFrame:
        """returns a breakdown by types"""
        pass

    def get_values_by_month(self, filters):
        """returns a breakdown by month"""
        pass
