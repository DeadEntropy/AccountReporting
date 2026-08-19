# coding=utf8
import glob
import os
import ast
import pandas as pd
import configparser

from concurrent.futures import ThreadPoolExecutor, as_completed

from bkanalysis.transforms.account_transforms import (
    barclays_transform as barc,
    clone_transform,
    citi_transform,
    lloyds_mortgage_transform as lloyds_mort,
    revolut_transform as rev_transform,
    lloyds_current_transform as lloyds_curr,
    nutmeg_isa_transform,
    nutmeg_transform,
    ubs_pension_transform,
    static_data as sd,
    vault_transform,
    coinbase_transform,
    coinbase_pro_transform,
    bnp_stock_transform,
    bnp_cash_transform,
    chase_transform,
    revolut_transform_2 as rev_transform_2,
    fidelity_transform,
    discover_transform,
    marcus_transform,
    discover_credit_transform,
    ubs_us_pension_transform,
    capital_one_transform,
    first_republic_transform,
    first_republic_mtg_transform,
    nutmeg_transaction_transform,
    ubs_wm_transform,
    script_transform,
    chasebusiness_transform,
    mortgage_script_transform,
)
from bkanalysis.config import config_helper as ch
from bkanalysis.market.market import Market
from bkanalysis.market import market_loader as ml

USE_LEGACY_LOADER = False


class Loader:
    GRANULAR_NUTMEG = True

    def __init__(self, config=None, include_market=True, ref_currency="GBP"):
        if config is None:
            self.config = configparser.ConfigParser()
            if len(self.config.read(ch.source)) != 1:
                raise OSError(f"no config found in {ch.source}")
        else:
            self.config = config

        self.market = None
        if "Market" in self.config and include_market:
            if "instr_to_preload" in self.config["Market"]:
                self.market = Market(
                    ml.MarketLoader().load(
                        ast.literal_eval(self.config["Market"]["instr_to_preload"]),
                        ref_currency,
                        "10y",
                    )
                )

        self.nutmeg_transformer = (
            ("Nutmeg", nutmeg_transform, self.market, ref_currency)
            if Loader.GRANULAR_NUTMEG
            else ("NutmegIsa", nutmeg_isa_transform, self.market, ref_currency)
        )

    def load_multi_thread(self, files):
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.load_internal, file): file for file in files}
            results = []
            for future in as_completed(futures):
                file = futures[future]
                try:
                    df = future.result()
                    df["SourceFile"] = os.path.basename(file)
                    results.append(df)
                except Exception as exc:
                    print(f"{file} generated an exception: {exc}")
        return pd.concat(results, ignore_index=True)

    def load_internal(self, file, ref_currency="GBP"):
        loaders = [
            ("Barclays", barc),
            ("LloydsCurrent", lloyds_curr),
            ("LloydsMortgage", lloyds_mort),
            self.nutmeg_transformer,
            ("Revolut", rev_transform, ";"),
            ("Revolut", rev_transform, ","),
            ("Revolut2", rev_transform_2, ","),
            ("Citi", citi_transform),
            ("UbsPension", ubs_pension_transform, self.market, ref_currency),
            ("UbsUsPension", ubs_us_pension_transform, None, ref_currency),
            ("Vault", vault_transform),
            ("CoinbasePro", coinbase_pro_transform),
            ("Coinbase", coinbase_transform),
            ("BnpStocks", bnp_stock_transform),
            ("BnpCash", bnp_cash_transform),
            ("Chase", chase_transform),
            ("ChaseBusiness", chasebusiness_transform),
            ("Fidelity", fidelity_transform),
            ("Discover", discover_transform),
            ("Marcus", marcus_transform),
            ("DiscoverCredit", discover_credit_transform),
            ("CapitalOne", capital_one_transform),
            ("FirstRepublic", first_republic_transform),
            ("FirstRepublicMortgage", first_republic_mtg_transform),
            # ('NutmegInvestment', nutmeg_transaction_transform),
            ("UbsWealthManagement", ubs_wm_transform),
            ("MortgageScript", mortgage_script_transform),
            ("Script", script_transform),
        ]

        for config_key, transform_module, *extra_args in loaders:
            if transform_module.can_handle(
                file,
                self.config[config_key] if config_key in self.config else None,
                *extra_args,
            ):
                return transform_module.load(
                    file,
                    self.config[config_key] if config_key in self.config else None,
                    *extra_args,
                )

        if clone_transform.can_handle(file):
            return clone_transform.load(file)

        raise ValueError(f"file {file} could not be processed by any of the loaders.")

    def load_old(self, file):
        df = self.load_internal_old(file)
        df["SourceFile"] = os.path.basename(file)
        return df

    def load_internal_old(self, file, ref_currency="GBP"):
        # (config section, module, can_handle extra args, load extra args);
        # a None section means the loader takes no config
        loaders = [
            ("Barclays", barc, (), ()),
            ("LloydsCurrent", lloyds_curr, (), ()),
            ("LloydsMortgage", lloyds_mort, (), ()),
            ("Revolut", rev_transform, (";",), (";",)),
            ("Revolut", rev_transform, (",",), (",",)),
            ("Revolut2", rev_transform_2, (",",), (",",)),
            ("Citi", citi_transform, (), ()),
            (None, clone_transform, (), ()),
            ("UbsPension", ubs_pension_transform, (), (self.market, ref_currency)),
            ("UbsUsPension", ubs_us_pension_transform, (), (None, ref_currency)),
            ("Vault", vault_transform, (), ()),
            ("CoinbasePro", coinbase_pro_transform, (), ()),
            ("Coinbase", coinbase_transform, (), ()),
            ("BnpStocks", bnp_stock_transform, (), ()),
            ("BnpCash", bnp_cash_transform, (), ()),
            ("Chase", chase_transform, (), ()),
            ("ChaseBusiness", chasebusiness_transform, (), ()),
            ("Fidelity", fidelity_transform, (), ()),
            ("Discover", discover_transform, (), ()),
            ("Marcus", marcus_transform, (), ()),
            ("DiscoverCredit", discover_credit_transform, (), ()),
            ("CapitalOne", capital_one_transform, (), ()),
            ("FirstRepublic", first_republic_transform, (), ()),
            ("FirstRepublicMortgage", first_republic_mtg_transform, (), ()),
            ("NutmegInvestment", nutmeg_transaction_transform, (), ()),
            ("UbsWealthManagement", ubs_wm_transform, (), ()),
            (None, mortgage_script_transform, (), ()),
            (None, script_transform, (), ()),
        ]

        for section, module, can_args, load_args in loaders:
            if section is not None and section not in self.config:
                continue
            config = self.config[section] if section is not None else None
            if module.can_handle(file, config, *can_args):
                return module.load(file, config, *load_args)

        raise ValueError(f"file {file} could not be processed by any of the loaders.")

    @staticmethod
    def get_files(folder_lake, root=None, include_xls=True, include_json=True):
        print(f"Loading files from {os.path.abspath(folder_lake)}")
        base = folder_lake if root is None else os.path.join(root, folder_lake)

        def find(extension):
            # glob is case-sensitive on Linux but case-insensitive on Windows, so look up
            # both the lowercase (e.g. *.csv) and uppercase (e.g. *.CSV) extensions and
            # de-duplicate the result to cover either platform.
            lower_case = glob.glob(os.path.join(base, f"*.{extension.lower()}"))
            upper_case = glob.glob(os.path.join(base, f"*.{extension.upper()}"))
            return sorted(set(lower_case) | set(upper_case))

        csv_files = find("csv")
        xls_files = find("xls") if include_xls else []
        json_files = find("json") if include_json else []

        print(f"Loading {len(csv_files)} CSV file(s), {len(xls_files)} XLS file(s), and {len(json_files)} JSON file(s).")
        return csv_files + xls_files + json_files

    def load_all(self, include_xls=True, include_json=True):
        if "folder_root" in self.config["IO"]:
            files = self.get_files(
                self.config["IO"]["folder_lake"],
                self.config["IO"]["folder_root"],
                include_xls,
                include_json,
            )
        else:
            files = self.get_files(
                self.config["IO"]["folder_lake"],
                include_xls=include_xls,
                include_json=include_json,
            )
        if len(files) == 0:
            return
        if USE_LEGACY_LOADER:
            df_list = [self.load_old(f) for f in files]
            for df_temp in df_list:
                df_temp["count"] = df_temp.groupby(sd.target_columns).cumcount()
            df = pd.concat(df_list)
            df = df.drop_duplicates().drop(["count"], axis=1).sort_values(["Date", "Account"], ascending=False)
        else:
            df = self.load_multi_thread(files)
        df.Date = pd.to_datetime(df.Date)
        df = df.sort_values(["Date", "Account"], ascending=False)
        return df.reset_index(drop=True)

    def save(self, df):
        df_copy = df.copy(True)
        df_copy.Date = df_copy.Date.apply(lambda x: x.strftime("%d-%b-%Y"))
        df_copy.to_csv(self.config["IO"]["path_aggregated"], index=False)

    def load_save(self):
        df = self.load_all()
        self.save(df)
