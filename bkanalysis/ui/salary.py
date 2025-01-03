from datetime import datetime
from bkanalysis.ui import ui

import pandas as pd


class Salary:
    """Class to handle Salaries"""

    OTHER_PAYROLLS = "Other Payrolls"

    def __prepare_scalar(self, year, base_payrolls_1, base_payrolls_2):
        self.payrolls = [base_payrolls_1, base_payrolls_2, Salary.OTHER_PAYROLLS]
        prev_year_end = f"{year-1}-12"

        self.outstanding_salaries = {
            base_payrolls_1: -self.monthly_salaries.GAP_1.sum(),
            base_payrolls_2: -self.monthly_salaries.GAP_2.sum(),
            Salary.OTHER_PAYROLLS: 0.0,
        }
        self.outstanding_salary = sum(
            [self.outstanding_salaries[p] for p in self.payrolls]
        )

        self.total_received_salaries = {
            p: self.monthly_salaries.loc[f"{year}":f"{year}-12"][p].sum()
            for p in self.payrolls
        }
        self.total_received_salary = sum(
            [self.total_received_salaries[p] for p in self.payrolls]
        )

        self.total_received_salaries_from_previous_year = {
            base_payrolls_1: -self.monthly_salaries.loc[prev_year_end].OUT_SLRY_1,
            base_payrolls_2: -self.monthly_salaries.loc[prev_year_end].OUT_SLRY_2,
            Salary.OTHER_PAYROLLS: 0.0,
        }
        self.total_received_salary_from_previous_year = sum(
            [self.total_received_salaries_from_previous_year[p] for p in self.payrolls]
        )

        self.actual_salaries = {
            p: self.total_received_salaries[p]
            - self.total_received_salaries_from_previous_year[p]
            for p in self.payrolls
        }
        self.actual_salary = sum([self.actual_salaries[p] for p in self.payrolls])

    def __init__(
        self,
        df_exp: pd.DataFrame,
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
        assert (
            anchor_date.year < year
        ), "anchor date is within the selected year, it must be before."
        self.anchor_date = anchor_date

        df_exp["MONTH"] = df_exp.Date.dt.strftime("%Y-%m")
        self.monthly_salaries = (
            pd.DataFrame(
                pd.pivot_table(
                    df_exp[
                        (df_exp.FullType == "Salary")
                        & (df_exp.Date > anchor_date)
                        & ~df_exp.MemoMapped.isin(exclude)
                    ],
                    values=ui.AMOUNT_CCY,
                    index="MONTH",
                    columns="MemoMapped",
                    aggfunc=sum,
                ).to_records()
            )
            .fillna(0)
            .set_index("MONTH")
        )

        self.other_payrolls = [
            p for p in self.monthly_salaries.columns if p not in payrolls_1 + payrolls_2
        ]

        self.monthly_salaries[base_payroll_1] = self.monthly_salaries[payrolls_1].sum(
            axis=1
        )
        self.monthly_salaries[base_payroll_2] = self.monthly_salaries[payrolls_2].sum(
            axis=1
        )
        self.monthly_salaries[Salary.OTHER_PAYROLLS] = self.monthly_salaries[
            self.other_payrolls
        ].sum(axis=1)

        self.monthly_salaries = self.monthly_salaries[
            [base_payroll_1, base_payroll_2, Salary.OTHER_PAYROLLS]
        ]

        self.monthly_salaries["GAP_1"] = (
            (self.monthly_salaries[base_payroll_1] - base_salary_1)
            if base_salary_1 is not None
            else 0.0
        )
        self.monthly_salaries["GAP_2"] = (
            (self.monthly_salaries[base_payroll_2] - base_salary_2)
            if base_salary_2 is not None
            else 0.0
        )
        self.monthly_salaries["OUT_SLRY_1"] = self.monthly_salaries.GAP_1.cumsum()
        self.monthly_salaries["OUT_SLRY_2"] = self.monthly_salaries.GAP_2.cumsum()

        self.__prepare_scalar(year, base_payroll_1, base_payroll_2)
