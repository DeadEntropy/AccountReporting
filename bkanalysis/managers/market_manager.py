from datetime import datetime
import pandas as pd

from bkanalysis.market import market_loader
from bkanalysis.managers.data_manager import DataManager


from bkanalysis.managers.manager_helper import normalize_date_column


class MarketManager:
    """Class to handle the interaction of the data with the market prices"""

    def __init__(self, ref_currency: str, config=None):
        self.config = config
        self.ref_currency = ref_currency
        self.prices = None
        self.asset_map = None

    def to_disk(self, path: str) -> None:
        """saves the transactions to disk"""
        self.prices.to_csv(path, index=True)
        pd.Series(self.asset_map, name="AssetMap").to_csv(f"{path.replace('.csv', '_asset_map.csv')}")

    def load_pregenerated_data(self, path) -> None:
        """loads the pregenerated data from disk"""
        self.prices = pd.read_csv(path, parse_dates=["Date"])
        self.prices = self.prices.set_index(["AssetMapped", "Date"])
        self.asset_map = pd.read_csv(f"{path.replace('.csv', '_asset_map.csv')}", index_col=0)["AssetMap"].to_dict()

    def get_asset_map(self, data_manager: DataManager) -> dict:
        """returns a map of the assets to their market symbol"""
        if self.asset_map is None:
            self.load_asset_map(data_manager)

        return self.asset_map

    def load_asset_map(self, data_manager: DataManager) -> None:
        """returns a map of the assets to their market symbol"""
        loader = market_loader.MarketLoader()
        self.asset_map = {
            instr: (loader.get_symbol(instr, "USD") if instr not in loader.source_map.keys() else instr)
            for instr in data_manager.transactions.Asset.dropna().unique()
        }

    def load_prices(self, data_manager: DataManager) -> None:
        """loads market prices from sources specified in the config file"""

        loader = market_loader.MarketLoader(self.config)
        assets = data_manager.assets
        period = {a: "10y" if (datetime.today() - assets[a]).days / 356 < 10 else "max" for a in assets}
        market_prices = loader.load(assets, self.ref_currency, period)

        df_prices = MarketManager.dataframe_from_nested_dict(market_prices)
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

        df["AssetPriceChangeInRefCurrency"] = -df.groupby("AssetMapped")["AssetPriceInRefCurrency"].diff()

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
