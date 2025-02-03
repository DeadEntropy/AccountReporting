from bkanalysis.transforms import master_transform
from bkanalysis.process import iat_identification
import tests.unit.config_helper as ch
import unittest
import pandas as pd
import datetime
import ast


class TestProcess(unittest.TestCase):

    def test_map_iat(self):
        config = ch.get_config()
        iat = iat_identification.IatIdentification(config)

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

        df_out = iat.map_iat(df)
        self.assertEqual(df_out.FacingAccount[0], "account2", "Intra-Account Transfer not correctly identified.")
        self.assertEqual(df_out.FacingAccount[1], "account1", "Intra-Account Transfer not correctly identified.")


if __name__ == "__main__":
    unittest.main()
