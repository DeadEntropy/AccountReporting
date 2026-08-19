# coding=utf8
import ast
import configparser
import pandas as pd
from bkanalysis.config import config_helper as ch


class IatIdentification:
    _use_old = True
    iat_types = ["SA", "IAT", "W_IN", "W_OUT", "SC", "R", "MC", "O", "FR", "TAX", "FPC", "FLC", "FLL", "FSC"]
    iat_full_types = [
        "Savings",
        "Intra-Account Transfert",
        "Wire In",
        "Wire Out",
        "Service Charge",
        "Rent/Mortgage",
        "Others",
        "Tax",
        "Flat Capital",
        "Flat Living Cost",
    ]
    iat_fx_types = ["FX"]
    relative_tolerance: float = 0.0

    def __init__(self, config=None):
        if config is None:
            self.config = configparser.ConfigParser()
            if len(self.config.read(ch.source)) != 1:
                raise OSError(f"no config found in {ch.source}")
        else:
            self.config = config

    def map_iat(self, df, iat_value_col="Amount", adjust_dates: bool = False):
        # Ensure columns match the expected structure
        columns_req = ["Currency", "FullType", "Date", "Account"] + [iat_value_col]
        assert all(
            [col_req in df.columns for col_req in columns_req]
        ), f"columns do not match expectation. Expected: {columns_req} but received: {df.columns}"

        # Ensure Date column is in datetime format
        df["Date"] = pd.to_datetime(df["Date"])

        # Create a new column for the offsetting account
        df["FacingAccount"] = None

        # Create a dictionary to store potential matches
        transaction_dict = {}
        df_mini = df[df.FullType.isin(self.iat_full_types)]
        for i, row in df_mini.iterrows():
            target_key = (row["Currency"], row["FullType"], -row[iat_value_col])
            key_this_row = (row["Currency"], row["FullType"], row[iat_value_col])
            date = row["Date"]

            # Check if there is a matching transaction in the dictionary
            if target_key in transaction_dict:
                for match_index in transaction_dict[target_key]:
                    if abs((date - df.loc[match_index, "Date"]).days) < 7 and df.loc[match_index, "FacingAccount"] is None:
                        df.loc[i, "FacingAccount"] = df.loc[match_index, "Account"]
                        df.loc[match_index, "FacingAccount"] = row["Account"]
                        if adjust_dates:
                            adjusted_date = max(df.loc[match_index, "Date"], row["Date"])
                            df.loc[match_index, "Date"] = adjusted_date
                            df.loc[i, "Date"] = adjusted_date
                        transaction_dict[target_key].remove(match_index)
                        break
            if key_this_row in transaction_dict:
                transaction_dict[key_this_row].append(i)
            else:
                transaction_dict[key_this_row] = [i]

        return df

    def map_iat_fx(self, df):
        new_columns = [n.strip() for n in ast.literal_eval(self.config["Mapping"]["new_columns"])]
        assert set(list(df.columns)) == set(new_columns), "columns do not match expectation."

        df["IDX"] = df.index
        df_mini = df[df.Type.str.upper().isin(self.iat_fx_types)][["AccountType", "Currency", "Type", "IDX", "Date", "Account", "Amount"]]
        df_mini["dummy_key"] = 1
        ndf = pd.merge(left=df_mini, right=df_mini, how="inner", on="dummy_key").drop("dummy_key", axis=1)
        out = ndf[
            (abs(ndf.Date_x - ndf.Date_y) < pd.Timedelta(7, "D"))
            & (ndf.Account_x != ndf.Account_y)
            & (ndf.Currency_x != ndf.Currency_y)
            & (ndf.Amount_x != 0)
        ]
        iat_transfers = list(zip(list(out.IDX_x.values), list(out.IDX_y.values)))
        df.drop("IDX", axis=1, inplace=True)

        offsetting_rows = set()
        for dup in iat_transfers:
            offsetting_rows.update(dup)

        for dup in iat_transfers:
            df.at[dup[0], "FacingAccount"] = df.loc[dup[1], "Account"]
            df.at[dup[1], "FacingAccount"] = df.loc[dup[0], "Account"]

        return df
