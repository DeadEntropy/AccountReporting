"""
class to manage salary
"""

from datetime import datetime

import pandas as pd


def create_legacy(transformation_manager):
    DEFAULT_PAYROLLS_1 = [
        "NAYA BIOSCIENCES PAYROLL",
        "CYTOVIA THERAPEUTICS",
        "NAYA ONCOLOGY",
        "TRINET HR CORPORATE PAYROLL",
    ]
    BASE_PAYROLL_1 = "NAYA_CYTOVIA"
    DEFAULT_PAYROLLS_2 = ["UBS SALARY", "UBS BONUS"]
    BASE_PAYROLL_2 = "UBS"
    EXCLUDE_DEFAULT = []
    BASE_SALARY_1 = 13250

    BASE_SALARY = {**{2024: BASE_SALARY_1}}

    return SalaryLegacy(
        transformation_manager,
        2024,
        datetime(2024 - 1, 1, 1),
        BASE_SALARY[2024],
        DEFAULT_PAYROLLS_1,
        BASE_PAYROLL_1,
        None,
        DEFAULT_PAYROLLS_2,
        BASE_PAYROLL_2,
        EXCLUDE_DEFAULT,
    )


def create_default(transformation_manager):
    salary_config = {
        "NAYA_CYTOVIA": {
            "base_salary": 13250,
            "payrolls": [
                "NAYA BIOSCIENCES PAYROLL",
                "CYTOVIA THERAPEUTICS",
                "NAYA ONCOLOGY",
                "TRINET HR CORPORATE PAYROLL",
            ],
        },
        "UBS": {
            "base_salary": None,
            "payrolls": ["UBS SALARY", "UBS BONUS"],
        },
    }
    exclude_default = []

    return Salary(
        transformation_manager,
        2024,
        datetime(2023, 1, 1),
        salary_config,
        exclude_default,
    )


class Salary:
    """Class to handle multiple salaries"""

    OTHER_PAYROLLS = "Other Payrolls"
    AMOUNT_CCY = "AmountCcy"

    def __prepare_scalar(self, year):
        self.payrolls = list(self.salaries_info.keys()) + [Salary.OTHER_PAYROLLS]
        prev_year_end = f"{year-1}-12"

        self.outstanding_salaries = {
            payroll: -self.monthly_salaries.loc[f"{year}":f"{year}-12"][f"GAP_{payroll}"].sum() for payroll in self.salaries_info.keys()
        }
        self.outstanding_salaries[Salary.OTHER_PAYROLLS] = 0.0
        self.outstanding_salary = sum(self.outstanding_salaries.values())

        self.total_received_salaries = {
            payroll: self.monthly_salaries.loc[f"{year}":f"{year}-12"][payroll].sum() for payroll in self.payrolls
        }
        self.total_received_salary = sum(self.total_received_salaries.values())

        self.total_received_salaries_from_previous_year = {
            payroll: -self.monthly_salaries.loc[prev_year_end][f"OUT_SLRY_{payroll}"] for payroll in self.salaries_info.keys()
        }
        self.total_received_salaries_from_previous_year[Salary.OTHER_PAYROLLS] = 0.0
        self.total_received_salary_from_previous_year = sum(self.total_received_salaries_from_previous_year.values())

        self.actual_salaries = {
            payroll: self.total_received_salaries[payroll] - self.total_received_salaries_from_previous_year[payroll]
            for payroll in self.payrolls
        }
        self.actual_salary = sum(self.actual_salaries.values())

    def __init__(
        self,
        transformation_manager,
        year: int = 2024,
        anchor_date: datetime = datetime(2023, 1, 1),
        salaries_info: dict = None,
        exclude: list = None,
    ):
        assert anchor_date.year < year, "anchor date is within the selected year, it must be before."
        self.anchor_date = anchor_date

        if exclude is None:
            exclude = []

        df_fl = transformation_manager.get_flow_values()
        df_salary = df_fl[df_fl.Type == "S"].reset_index()
        df_salary = df_salary[df_salary.Date > anchor_date]
        df_salary["MONTH"] = df_salary.Date.dt.strftime("%Y-%m")
        self.monthly_salaries = (
            pd.DataFrame(
                pd.pivot_table(
                    df_salary[~df_salary.MemoMapped.isin(exclude)],
                    values="Value",
                    index="MONTH",
                    columns="MemoMapped",
                    aggfunc="sum",
                ).to_records()
            )
            .fillna(0)
            .set_index("MONTH")
        )

        self.salaries_info = salaries_info or {}

        # Filter relevant payrolls
        for payroll, info in self.salaries_info.items():
            relevant_payrolls = [p for p in info["payrolls"] if p in self.monthly_salaries.columns]
            self.monthly_salaries[payroll] = self.monthly_salaries[relevant_payrolls].sum(axis=1)
            self.monthly_salaries[f"GAP_{payroll}"] = (
                (self.monthly_salaries[payroll] - info["base_salary"]) if info["base_salary"] is not None else 0.0
            )
            self.monthly_salaries[f"OUT_SLRY_{payroll}"] = self.monthly_salaries[f"GAP_{payroll}"].cumsum()

        # Determine other payrolls
        explicit_payrolls = [x for xs in [y["payrolls"] for y in self.salaries_info.values()] for x in xs]
        self.other_payrolls = [p for p in df_salary.MemoMapped.unique() if p not in explicit_payrolls and p not in exclude]
        self.monthly_salaries[Salary.OTHER_PAYROLLS] = self.monthly_salaries[self.other_payrolls].sum(axis=1)

        self.__prepare_scalar(year)


class SalaryLegacy:
    """Legacy Class to handle Salaries"""

    OTHER_PAYROLLS = "Other Payrolls"
    AMOUNT_CCY = "AmountCcy"

    def __prepare_scalar(self, year, base_payrolls_1, base_payrolls_2):
        self.payrolls = [base_payrolls_1, base_payrolls_2, SalaryLegacy.OTHER_PAYROLLS]
        prev_year_end = f"{year-1}-12"

        self.outstanding_salaries = {
            base_payrolls_1: -self.monthly_salaries.GAP_1[f"{year}":f"{year}-12"].sum(),
            base_payrolls_2: -self.monthly_salaries.GAP_2[f"{year}":f"{year}-12"].sum(),
            SalaryLegacy.OTHER_PAYROLLS: 0.0,
        }
        self.outstanding_salary = sum([self.outstanding_salaries[p] for p in self.payrolls])

        self.total_received_salaries = {p: self.monthly_salaries.loc[f"{year}":f"{year}-12"][p].sum() for p in self.payrolls}
        self.total_received_salary = sum([self.total_received_salaries[p] for p in self.payrolls])

        self.total_received_salaries_from_previous_year = {
            base_payrolls_1: -self.monthly_salaries.loc[prev_year_end].OUT_SLRY_1,
            base_payrolls_2: -self.monthly_salaries.loc[prev_year_end].OUT_SLRY_2,
            SalaryLegacy.OTHER_PAYROLLS: 0.0,
        }
        self.total_received_salary_from_previous_year = sum([self.total_received_salaries_from_previous_year[p] for p in self.payrolls])

        self.actual_salaries = {
            p: self.total_received_salaries[p] - self.total_received_salaries_from_previous_year[p] for p in self.payrolls
        }
        self.actual_salary = sum([self.actual_salaries[p] for p in self.payrolls])

    def __init__(
        self,
        transformation_manager,
        year: int = 2024,
        anchor_date: datetime = datetime(2023, 1, 1),
        base_salary_1: float = None,
        payrolls_1: list = None,
        base_payroll_1: str = None,
        base_salary_2: float = None,
        payrolls_2: list = None,
        base_payroll_2: str = None,
        exclude: list = None,
    ):
        assert anchor_date.year < year, "anchor date is within the selected year, it must be before."
        self.anchor_date = anchor_date

        if exclude is None:
            exclude = []

        df_fl = transformation_manager.get_flow_values()
        df_salary = df_fl[df_fl.Type == "S"].reset_index()
        df_salary = df_salary[df_salary.Date > anchor_date]
        df_salary["MONTH"] = df_salary.Date.dt.strftime("%Y-%m")
        self.monthly_salaries = (
            pd.DataFrame(
                pd.pivot_table(
                    df_salary[~df_salary.MemoMapped.isin(exclude)],
                    values="Value",
                    index="MONTH",
                    columns="MemoMapped",
                    aggfunc="sum",
                ).to_records()
            )
            .fillna(0)
            .set_index("MONTH")
        )

        payrolls_1 = [p for p in payrolls_1 if p in self.monthly_salaries.columns]
        payrolls_2 = [p for p in payrolls_2 if p in self.monthly_salaries.columns]

        self.other_payrolls = [p for p in self.monthly_salaries.columns if p not in payrolls_1 + payrolls_2]

        self.monthly_salaries[base_payroll_1] = self.monthly_salaries[payrolls_1].sum(axis=1)
        self.monthly_salaries[base_payroll_2] = self.monthly_salaries[payrolls_2].sum(axis=1)
        self.monthly_salaries[Salary.OTHER_PAYROLLS] = self.monthly_salaries[self.other_payrolls].sum(axis=1)

        self.monthly_salaries = self.monthly_salaries[[base_payroll_1, base_payroll_2, SalaryLegacy.OTHER_PAYROLLS]].copy()

        self.monthly_salaries["GAP_1"] = (self.monthly_salaries[base_payroll_1] - base_salary_1) if base_salary_1 is not None else 0.0
        self.monthly_salaries["GAP_2"] = (self.monthly_salaries[base_payroll_2] - base_salary_2) if base_salary_2 is not None else 0.0
        self.monthly_salaries["OUT_SLRY_1"] = self.monthly_salaries.GAP_1.cumsum()
        self.monthly_salaries["OUT_SLRY_2"] = self.monthly_salaries.GAP_2.cumsum()

        self.__prepare_scalar(year, base_payroll_1, base_payroll_2)
