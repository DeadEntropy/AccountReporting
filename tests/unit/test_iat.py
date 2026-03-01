"""Tests for inter-account transfer identification."""
import ast
import datetime

import pandas as pd
import pytest

from bkanalysis.process import iat_identification


class TestProcess:
    """Tests for transaction processing."""

    def test_map_iat(self, config, iat_identifier):
        """Verify inter-account transfers are correctly identified and matched."""
        # Create a test dataframe with required columns
        df = pd.DataFrame(columns=[n.strip() for n in ast.literal_eval(config["Mapping"]["new_columns"])])

        df["Account"] = ["account1", "account2", "account1"]
        df["Memo"] = ["memo"] * 3
        df["AccountType"] = ["account_type"] * 3
        df["Currency"] = ["CCY"] * 3
        df["Type"] = ["IAT"] * 3
        df["FullType"] = ["Intra-Account Transfert"] * 3
        df["Amount"] = [100, -100, 200]
        df["Date"] = datetime.datetime(2020, 10, 15)
        df.fillna("", inplace=True)

        # Process and verify matching
        df_out = iat_identifier.map_iat(df)
        assert df_out.FacingAccount[0] == "account2", "Intra-Account Transfer not correctly identified."
        assert df_out.FacingAccount[1] == "account1", "Intra-Account Transfer not correctly identified."
