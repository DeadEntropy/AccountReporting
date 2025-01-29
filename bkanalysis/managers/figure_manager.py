from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from bkanalysis.process.iat_identification import IatIdentification
from bkanalysis.salary import Salary

from bkanalysis.managers.transformation_manager import TransformationManager


class FigureManager:
    """Class to prepare the data for the UI"""

    __DATE_FORMAT = "%Y-%m-%d"

    def __init__(self, transformation_manager: TransformationManager):
        self.transformation_manager = transformation_manager
        self._iat_types = IatIdentification.iat_types + ["C", "S"]

    def __prepare_timeseries_label(self, labels, max_labels: int = 20, max_char: int = 20):
        """prepares the label for the hover"""
        total_amt = sum([l[1] for l in labels])
        total_label = f"<br><br>TOTAL: {total_amt:,.0f}" if total_amt != 0 else ""

        if len(labels) < max_labels:
            try:
                return "<br>".join([f"{v:>7,.0f}: {k[:max_char] if isinstance(k,str) else ''}" for (k, v, t, s) in labels]) + total_label
            except TypeError:
                return ""

        threshold = sorted([abs(v) for (k, v, t, s) in labels], reverse=True)[max_labels - 1]
        largest_labels = [(k, v) for (k, v, t, s) in labels if abs(v) > threshold]
        remainging_amount = sum([v for k, v, t, s in labels]) - sum([v for k, v in largest_labels])
        largest_labels = sorted(
            {**dict(largest_labels), **{"OTHER": remainging_amount}}.items(),
            key=lambda item: -abs(item[1]),
        )
        return "<br>".join([f"{v:>7,.0f}: {k[:25]}" for (k, v) in largest_labels]) + total_label

    def __get_timeseries_annotations(self, labels: pd.Series, annotation_count: int = 3) -> list:
        """returns the annotations for the figure"""
        result = [(index, *tup) for index, sublist in labels.items() for tup in sublist if tup[2] not in self._iat_types]
        df_annotations = (
            pd.DataFrame(result, columns=["Date", "MemoMapped", "TransactionValue", "Type", "SubType"])
            .sort_values(by="TransactionValue", ascending=True)
            .iloc[:annotation_count]
            .reset_index(drop=True)
        )

        df_annotations["Annotation"] = [
            f"{memo[:20]}: {tv:,.0f}" for memo, tv in zip(df_annotations["MemoMapped"], df_annotations["TransactionValue"])
        ]

        return df_annotations.groupby("Date").agg({"Annotation": lambda x: "<br>".join(x)})

    def get_figure_timeseries(self, date_range: list = None, account: str | list = None, annotation_count: int = 3) -> go.Figure:
        """plots the timeseries of the account value"""
        df_values_timeseries = self.transformation_manager.get_values_timeseries(date_range, account)
        df_annotations = self.__get_timeseries_annotations(df_values_timeseries["TransactionValue_list"], annotation_count=annotation_count)

        fig = go.Figure(
            data=go.Scatter(
                x=df_values_timeseries.index,
                y=df_values_timeseries["Value"],
                hovertext=df_values_timeseries["TransactionValue_list"].apply(self.__prepare_timeseries_label),
            ),
        )

        fig.update_layout(
            title="Total Wealth",
            xaxis_title="Date",
            yaxis_title="Currency",
            margin=dict(t=50),
        )

        i = 0
        for annotation_date in df_annotations.index:
            ann_text = df_annotations.loc[annotation_date]["Annotation"]
            fig.add_annotation(
                x=annotation_date,
                y=df_values_timeseries.loc[annotation_date]["Value"],
                xref="x",
                yref="y",
                text=ann_text,
                showarrow=True,
                font=dict(family="Courier New, monospace", size=12, color="#ffffff"),
                align="center",
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#636363",
                ax=20,
                ay=(-100 if (i % 4) == 0 else 50 if (i % 3) == 0 else -50 if (i % 2) == 0 else 100),
                bordercolor="#c7c7c7",
                borderwidth=2,
                borderpad=4,
                bgcolor="#ff7f0e",
                opacity=0.65,
            )
            i = i + 1

        return fig

    def get_total_wealth(self, date: datetime = None) -> float:
        """returns the total wealth at the given date"""
        df_values_timeseries = self.transformation_manager.get_values_timeseries()
        if date is None:
            return df_values_timeseries.iloc[-1]["Value"]
        return df_values_timeseries.loc[date]["Value"]

    @staticmethod
    def __get_title_sunburst(dates, amounts):
        """returns the title of the sunburst chart"""
        if dates is not None:
            return (
                f"Spending Breakdown for {min(dates).strftime(FigureManager.__DATE_FORMAT)} to "
                f"{max(dates).strftime(FigureManager.__DATE_FORMAT)}"
                f" (Total Spend: {amounts.sum():,.0f})"
            )
        return f" (Total Spend: {amounts.sum():,.0f})"

    def get_figure_sunburst(self, date_range: list = None, account: str = None, include_iat=False) -> go.Figure:
        """plots a sunburst of the account transactions"""
        df_expenses = self.transformation_manager.get_flow_values(
            date_range[0], date_range[1], account, how="out", include_iat=include_iat
        ).reset_index(drop=True)
        df_expenses["Value"] = (-1) * df_expenses["Value"]

        title = self.__get_title_sunburst(date_range, df_expenses["Value"])
        return px.sunburst(df_expenses, path=["FullType", "FullSubType", "MemoMapped"], values="Value", title=title)

    def get_figure_bar(
        self, category, label, show_count: int = 5, date_range: list = None, account: str = None, how: str = "out", include_iat=False
    ):
        """plots a bar chat showing the monthly spending by category"""
        df_expenses = self.transformation_manager.get_flow_values(date_range[0], date_range[1], account, how=how, include_iat=include_iat)
        df_expenses["Value"] = (-1) * df_expenses["Value"]
        key = list(category.keys())[0]

        if key not in df_expenses.columns:
            return px.bar(
                pd.DataFrame(columns=["Month", "MemoMapped", "Value"]),
                x="Month",
                y="Value",
                color="MemoMapped",
                text="MemoMapped",
                title=f"{category[key]} Spending",
            )

        df_expenses = df_expenses[df_expenses[key] == category[key]].reset_index()[["Date", label, "Value"]]

        if len(df_expenses) == 0:
            return px.bar(
                pd.DataFrame(columns=["Month", "MemoMapped", "Value"]),
                x="Month",
                y="Value",
                color="MemoMapped",
                text="MemoMapped",
                title=f"{category[key]} Spending",
            )

        df_expenses["Date"] = pd.to_datetime(df_expenses["Date"])
        df_expenses = df_expenses.groupby([df_expenses["Date"].dt.to_period("M"), label])["Value"].sum().reset_index()
        df_expenses = df_expenses.rename(columns={"Date": "Month"})
        df_expenses["Month"] = df_expenses["Month"].astype(str)

        category_totals = df_expenses.groupby(label)["Value"].sum().astype(float)

        if show_count is None:
            COVERAGE = 0.8
            # Calculate total sum of all categories
            total_sum = category_totals.sum()

            # Calculate cumulative sum and find the index for top categories that represent more than 80% of the total
            category_totals_sorted = category_totals.sort_values(ascending=False)
            cumulative_sum = category_totals_sorted.cumsum()

            cum_coverage = cumulative_sum[cumulative_sum <= total_sum * COVERAGE]
            if len(cum_coverage) > 10:
                top_n = cum_coverage.index[9]
            else:
                top_n = cum_coverage.index[-1]

            # Get the top categories
            top_categories = category_totals_sorted.loc[:top_n].index
        else:
            top_categories = category_totals.nlargest(show_count).index

        df_expenses[label] = df_expenses[label].apply(lambda x: x if x in top_categories else "Others")
        df_expenses_aggregated = df_expenses.groupby(["Month", label], as_index=False)["Value"].sum()

        return px.bar(
            df_expenses_aggregated, x="Month", y="Value", color="MemoMapped", text="MemoMapped", title=f"{category[key]} Spending"
        )

    def get_category_breakdown(
        self, category, label, row_limit: int = 5, date_range: list = None, account: str = None, how: str = "out", include_iat=False
    ):
        """Return the top categories based on the provided filter"""
        key = list(category.keys())[0]
        df_expenses = self.transformation_manager.get_flow_values(date_range[0], date_range[1], account, how=how, include_iat=include_iat)

        if key not in df_expenses.columns:
            return pd.DataFrame(columns=["MemoMapped", "Value"])

        df_expenses = df_expenses[df_expenses[key] == category[key]].reset_index()[["Date", label, "Value"]]

        if len(df_expenses) == 0:
            return pd.DataFrame(columns=["MemoMapped", "Value"])

        return (
            pd.DataFrame(pd.pivot_table(df_expenses, index=label, values="Value", aggfunc="sum").to_records())
            .sort_values("Value")
            .reset_index(drop=True)[:row_limit]
        )

    def get_figure_waterfall(
        self,
        date_range: list = None,
        account: str = None,
        how: str = "both",
        include_iat=False,
        salary_override: Salary = None,
        include_capital_gain: bool = True,
    ):
        """plots a waterfall chat showing the spending/incomes by category"""
        df_expenses = self.transformation_manager.get_flow_values(
            date_range[0], date_range[1], account, how=how, include_iat=include_iat, include_full_types=True
        )
        categorised_flows = df_expenses.reset_index(drop=True).groupby("FullType")["Value"].sum().sort_values(ascending=False)

        if salary_override is not None:
            categorised_flows = categorised_flows.drop("Salary", axis=0)
            for p in salary_override.payrolls:
                categorised_flows.loc[p] = salary_override.actual_salaries[p]
            categorised_flows.loc["Salary Carried Over"] = salary_override.total_received_salary_from_previous_year

        threshold = 2000
        small_flows = (
            categorised_flows[
                (categorised_flows > -threshold) & (categorised_flows < threshold) & (categorised_flows.index != "Others")
            ].sum()
            + categorised_flows.loc["Others"]
        )
        categorised_flows = categorised_flows[
            (categorised_flows < -threshold) | (categorised_flows > threshold) | (categorised_flows.index == "Others")
        ]
        categorised_flows.loc["Others"] = small_flows

        if include_capital_gain:
            categorised_flows["Capital Gain"] = self.transformation_manager.get_values_by_asset(date_range=date_range, account=account)[
                "CapitalGain"
            ].sum()

        categorised_flows = categorised_flows.sort_values(ascending=False)

        fig = go.Figure(
            go.Waterfall(
                name="20",
                orientation="v",
                measure=["relative" for b in categorised_flows] + ["total"],
                x=list(categorised_flows.index) + ["savings"],
                textposition="outside",
                text=list(categorised_flows.index) + ["Savings"],
                y=list(categorised_flows.values) + [-categorised_flows.values.sum()],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            )
        )

        title = f"Income/Spending Summary {date_range[1].year}" if date_range is not None else "Income/Spending Summary"
        fig.update_layout(showlegend=False, margin=dict(t=50), title=title)

        return fig
