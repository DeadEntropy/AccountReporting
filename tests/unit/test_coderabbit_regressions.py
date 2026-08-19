"""Regression tests for the fixes applied from the CodeRabbit full-codebase review."""
import configparser
import datetime
from datetime import datetime as dt

import numpy as np
import pandas as pd
import pytest

from bkanalysis.managers import manager_helper
from bkanalysis.market.market import Market
from bkanalysis.market.price import Price
from bkanalysis.process import process_helper
from bkanalysis.process.iat_identification import IatIdentification
from bkanalysis.process.status import LastUpdate
from bkanalysis.tax import tax
from bkanalysis.tax import nutmeg
from bkanalysis.transforms.account_transforms import citi_transform
from bkanalysis.transforms.account_transforms import mortgage_script_transform


class TestMarketClosestDate:
    """Interpolation must use the closest surrounding dates, not the last date in the series."""

    def test_linear_interpolation_uses_closest_following_date(self):
        market = Market(
            {
                "FOO": {
                    dt(2023, 1, 1): Price(100.0, "GBP"),
                    dt(2023, 1, 5): Price(200.0, "GBP"),
                    dt(2023, 12, 31): Price(1000.0, "GBP"),
                }
            },
            linear_interpolation=True,
        )
        # midway between 1 Jan (100) and 5 Jan (200); before the fix the "next"
        # date resolved to 31 Dec and the interpolated price was wrong
        assert market.get_price_in_currency("FOO", dt(2023, 1, 3), "GBP") == 150.0

    def test_price_just_after_first_date_does_not_raise(self):
        market = Market({"FOO": {dt(2023, 1, 1): Price(100.0, "GBP"), dt(2023, 1, 10): Price(200.0, "GBP")}})
        assert market.get_price_in_currency("FOO", dt(2023, 1, 2), "GBP") == 100.0


class TestTaxYearValidation:
    def test_unsupported_year_raises_value_error(self):
        with pytest.raises(ValueError, match="Available years"):
            tax.compute_uk_tax_return(50000, 0.0, tax_year="1999")


class TestNutmegPurchases:
    def test_zero_unit_sale_returns_empty_dict(self):
        assert nutmeg.get_relevant_purchases_for_sale(0, {dt(2020, 1, 1): 10.0}) == {}

    def test_remaining_purchases_with_no_relevant_purchases_is_a_no_op(self):
        purchases = {dt(2020, 1, 1): 10.0, dt(2020, 2, 1): 5.0}
        assert nutmeg.get_remaining_purchases({}, purchases) == purchases

    def test_insufficient_purchases_raise_value_error(self):
        with pytest.raises(ValueError, match="do not cover"):
            nutmeg.get_relevant_purchases_for_sale(100.0, {dt(2020, 1, 1): 10.0})


class TestCitiMemo:
    def test_transfer_to_money_market_label(self):
        assert citi_transform.simplify_memo("Transfer to Money Market 123") == "Transfer to Money Market"


class TestFiscalYear:
    def test_fifth_of_april_belongs_to_previous_fiscal_year(self):
        assert process_helper.get_fiscal_year(datetime.datetime(2023, 4, 5)) == 2022

    def test_sixth_of_april_belongs_to_current_fiscal_year(self):
        assert process_helper.get_fiscal_year(datetime.datetime(2023, 4, 6)) == 2023


class TestIsCcy:
    def test_non_string_input_returns_false(self):
        assert manager_helper.is_ccy(None) is False
        assert manager_helper.is_ccy(np.nan) is False

    def test_valid_pair_still_accepted(self):
        assert manager_helper.is_ccy("GBPUSD=X") is True


class TestLastUpdate:
    def test_supplied_config_is_kept(self):
        config = configparser.ConfigParser()
        config["IO"] = {"path_aggregated": "a.csv", "path_last_updated": "b.csv"}
        assert LastUpdate(config).config is config


class TestIatTolerance:
    @staticmethod
    def __make_df(amount_a, amount_b):
        return pd.DataFrame(
            {
                "Currency": ["GBP", "GBP"],
                "FullType": ["Savings", "Savings"],
                "Date": [dt(2023, 1, 1), dt(2023, 1, 2)],
                "Account": ["A", "B"],
                "Amount": [amount_a, amount_b],
            }
        )

    def test_exact_match_still_pairs(self):
        iat = IatIdentification(configparser.ConfigParser())
        df = iat.map_iat(self.__make_df(100.0, -100.0))
        assert df.loc[1, "FacingAccount"] == "A"
        assert df.loc[0, "FacingAccount"] == "B"

    def test_amounts_within_tolerance_pair(self):
        iat = IatIdentification(configparser.ConfigParser())
        iat.relative_tolerance = 0.01
        df = iat.map_iat(self.__make_df(100.0, -100.5))
        assert df.loc[1, "FacingAccount"] == "A"

    def test_amounts_outside_tolerance_do_not_pair(self):
        iat = IatIdentification(configparser.ConfigParser())
        df = iat.map_iat(self.__make_df(100.0, -100.5))
        assert df.loc[1, "FacingAccount"] is None

    def test_zero_amounts_do_not_pair(self):
        iat = IatIdentification(configparser.ConfigParser())
        df = iat.map_iat(self.__make_df(0.0, 0.0))
        assert df.loc[0, "FacingAccount"] is None
        assert df.loc[1, "FacingAccount"] is None


class TestMortgageScriptTransform:
    def test_can_handle_returns_false_for_malformed_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        assert mortgage_script_transform.can_handle(str(bad), None) is False

    def test_can_handle_returns_false_for_missing_file(self, tmp_path):
        assert mortgage_script_transform.can_handle(str(tmp_path / "missing.json"), None) is False

    def test_can_handle_returns_false_for_empty_or_array_json(self, tmp_path):
        empty = tmp_path / "empty.json"
        empty.write_text("{}")
        assert mortgage_script_transform.can_handle(str(empty), None) is False
        arr = tmp_path / "arr.json"
        arr.write_text("[1, 2]")
        assert mortgage_script_transform.can_handle(str(arr), None) is False

    def test_new_mortgage_returns_empty_frame_without_error(self):
        data = {
            "start_date": dt.now().strftime("%d-%b-%Y"),
            "interest": 0.035,
            "term": 25,
            "principal": 100000,
            "account_name": "MTG",
            "currency": "GBP",
        }
        df = mortgage_script_transform.get_cashflows(data)
        assert len(df) == 0
