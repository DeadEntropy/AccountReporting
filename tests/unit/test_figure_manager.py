"""Tests for the FigureManager class."""
import math
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go

from bkanalysis.managers import data_manager, market_manager, transformation_manager
from bkanalysis.managers.figure_manager import FigureManager


class _StubSalary:
    """Minimal salary override for the waterfall figure."""

    payrolls = ["ACME"]
    actual_salaries = {"ACME": 5000.0}
    total_received_salary_from_previous_year = 0.0


def _build_figure_manager_without_salary_flows(config):
    """FigureManager over transactions that contain no 'Salary' and no 'Others' FullType."""
    dm = data_manager.DataManager(config)
    dm.transactions = pd.DataFrame(
        {
            "Account": ["Account1"] * 4,
            "Asset": ["USD"] * 4,
            "Date": pd.date_range("2023-06-01", periods=4),
            "Quantity": [-3000.0, -150.0, -120.0, -80.0],
            "MemoMapped": ["RENT", "SHOP A", "SHOP B", "SHOP C"],
            "Type": ["R", "G", "G", "G"],
            "SubType": ["RNT", "GRO", "GRO", "GRO"],
            "FullType": ["Rent", "Grocery", "Grocery", "Grocery"],
            "FullSubType": ["Rent", "Grocery", "Grocery", "Grocery"],
            "FullMasterType": ["Living Costs"] * 4,
            "AccountType": ["current"] * 4,
        }
    )

    mm = market_manager.MarketManager(ref_currency="USD", config=config)
    mm.asset_map = {"USD": "USD"}
    mm.prices = pd.DataFrame(
        {
            "AssetMapped": ["USD"] * 4,
            "Date": pd.date_range("2023-06-01", periods=4),
            "AssetPriceInRefCurrency": [1.0] * 4,
            "AssetPriceChangeInRefCurrency": [0.0] * 4,
        }
    ).set_index(["AssetMapped", "Date"])

    tm = transformation_manager.TransformationManager(dm, mm)
    return FigureManager(tm)


class TestFigureManager:
    """Tests for FigureManager robustness."""

    def test_waterfall_without_salary_or_others_categories(self, config):
        """The waterfall must not raise when the period has no 'Salary' flow and no 'Others' bucket."""
        fm = _build_figure_manager_without_salary_flows(config)

        fig = fm.get_figure_waterfall(
            date_range=[datetime(2023, 1, 1), datetime(2023, 12, 31)],
            salary_override=_StubSalary(),
            include_capital_gain=False,
        )

        assert isinstance(fig, go.Figure)

    def test_get_saving_ratio_without_income_returns_nan(self):
        """A period with no income must return NaN instead of dividing by zero."""
        transactions = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2023-03-01", "2023-04-01"]),
                "Quantity": [-100.0, -200.0],
                "FullType": ["Grocery", "Rent"],
                "FullSubType": ["Grocery", "Rent"],
                "FullMasterType": ["Living Costs", "Living Costs"],
            }
        )
        fm = FigureManager(SimpleNamespace(data_manager=SimpleNamespace(transactions=transactions)))

        assert math.isnan(fm.get_saving_ratio(2023))
