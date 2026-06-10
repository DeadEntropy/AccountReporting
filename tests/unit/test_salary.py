"""Tests for the salary module."""
from datetime import datetime

import pandas as pd

from bkanalysis.salary import Salary, SalaryLegacy


class _StubTransformationManager:
    """Minimal stand-in returning a fixed flow-values DataFrame."""

    def __init__(self, df_flow):
        self._df_flow = df_flow

    def get_flow_values(self):
        return self._df_flow


def _flow_values_without_previous_december():
    """Salary flows only inside 2024: no row exists for 2023-12."""
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"]),
            "Type": ["S", "S", "S"],
            "MemoMapped": ["ACME PAYROLL"] * 3,
            "Value": [900.0, 900.0, 900.0],
        }
    )


class TestSalary:
    """Tests for the Salary class."""

    def test_no_salary_in_previous_december(self):
        """A missing previous-December row must not raise and carries over 0 salary."""
        tm = _StubTransformationManager(_flow_values_without_previous_december())

        salary = Salary(
            tm,
            2024,
            datetime(2023, 1, 1),
            {"ACME": {"base_salary": 1000, "payrolls": ["ACME PAYROLL"]}},
            [],
        )

        assert salary.total_received_salary_from_previous_year == 0.0
        assert salary.actual_salary == 2700.0


class TestSalaryLegacy:
    """Tests for the SalaryLegacy class."""

    def test_no_salary_in_previous_december(self):
        """A missing previous-December row must not raise and carries over 0 salary."""
        tm = _StubTransformationManager(_flow_values_without_previous_december())

        salary = SalaryLegacy(
            tm,
            2024,
            datetime(2023, 1, 1),
            1000,
            ["ACME PAYROLL"],
            "ACME",
            None,
            ["UNKNOWN PAYROLL"],
            "OTHER_BASE",
            [],
        )

        assert salary.total_received_salary_from_previous_year == 0.0
        assert salary.actual_salary == 2700.0
