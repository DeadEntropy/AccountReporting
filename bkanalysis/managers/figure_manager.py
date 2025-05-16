from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from bkanalysis.process.iat_identification import IatIdentification
from bkanalysis.salary import Salary

from bkanalysis.managers.transformation_manager import TransformationManager
from bkanalysis.managers.manager_helper import is_ccy


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

    def get_figure_sunburst(self, date_range: list = None, account: str = None, include_iat=False, how="out") -> go.Figure:
        """plots a sunburst of the account transactions"""
        df_expenses = self.transformation_manager.get_flow_values(
            date_range[0], date_range[1], account, how=how, include_iat=include_iat
        ).reset_index(drop=True)
        df_expenses["Value"] = (-1) * df_expenses["Value"]

        path = ["FullType", "FullSubType", "MemoMapped"]
        df_expenses = pd.pivot_table(df_expenses, values="Value", index=path, aggfunc="sum").reset_index()
        df_expenses = df_expenses[df_expenses.Value > 0]
        df_expenses["formatted_value"] = df_expenses["Value"].apply(lambda x: f"${x:,.0f}" if x < 10000 else f"${x/1000:,.0f}K")

        title = self.__get_title_sunburst(date_range, df_expenses["Value"])
        fig = px.sunburst(df_expenses, path=path, values="Value", title=title)

        # Add custom hovertemplate
        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>Value: %{value:$,.0f}",
        )

        return fig

    def get_figure_bar(
        self,
        category,
        label,
        show_count: int = 5,
        date_range: list = None,
        account: str = None,
        how: str = "out",
        include_iat=False,
        return_bar: bool = True,
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

        top_categories = self.get_top_categories(category_totals, show_count)

        df_expenses[label] = df_expenses[label].apply(lambda x: x if x in top_categories else "Others")
        df_expenses_aggregated = df_expenses.groupby(["Month", label], as_index=False)["Value"].sum()

        if return_bar:
            fig = px.bar(
                df_expenses_aggregated, x="Month", y="Value", color="MemoMapped", text="MemoMapped", title=f"{category[key]} Spending"
            )

            # Add custom hovertemplate
            fig.update_traces(
                hovertemplate="<b>%{text}</b><br>Value: %{value:$,.0f}",
            )

            return fig
        else:
            # Aggregate expenses by Month and Category
            df_expenses_aggregated = df_expenses.groupby(["Month", label], as_index=False)["Value"].sum()

            # Compute total sum per category and sort columns by total value (descending)
            category_totals = df_expenses_aggregated.groupby("MemoMapped")["Value"].sum()
            sorted_categories = category_totals.sort_values(ascending=False).index.tolist()

            # Compute stacked totals with sorted order
            stacked_totals = df_expenses_aggregated.pivot(index="Month", columns="MemoMapped", values="Value").fillna(0)
            stacked_totals = stacked_totals[sorted_categories]  # Enforce largest-to-smallest order
            stacked_totals = stacked_totals.cumsum(axis=1)  # Compute cumulative stacked values

            # Reorder df_expenses_aggregated to match sorted order
            df_expenses_aggregated["MemoMapped"] = pd.Categorical(
                df_expenses_aggregated["MemoMapped"], categories=sorted_categories, ordered=True
            )

            # Create stacked line chart
            fig = px.line(
                df_expenses_aggregated,
                x="Month",
                y="Value",
                color="MemoMapped",
                title=f"{category[key]} Spending",
                line_group="MemoMapped",
                category_orders={"MemoMapped": sorted_categories},  # Enforce order in the plot
                text="MemoMapped",
            )

            # Enable stacking
            fig.update_traces(
                mode="lines+markers",
                stackgroup="one",
                hovertemplate="<b>%{text}: %{y:$,.0f}<extra></extra>",
            )

            # Adjust label positioning
            for memo in sorted_categories:  # Iterate in sorted order
                df_memo = df_expenses_aggregated[df_expenses_aggregated["MemoMapped"] == memo]

                # Find the month where this category has the maximum value
                max_idx = df_memo["Value"].idxmax()
                max_month = df_memo.loc[max_idx, "Month"]

                # Get the stacked value at that month
                stacked_value = stacked_totals.loc[max_month, memo]
                y_shift = -0.025 * stacked_value

                # Find the corresponding color from the figure traces
                trace_color = None
                for trace in fig.data:
                    if trace.name == memo:
                        trace_color = trace.line.color
                        break  # Stop searching once we find it

                fig.add_annotation(
                    x=max_month,  # Shift right
                    y=stacked_value + y_shift,  # Shift below
                    text=memo,
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    xanchor="center",
                    yanchor="top",  # Ensures label is always below the point,
                    bgcolor="rgba(0,0,0,0)",  # Set background color to match line
                    bordercolor=trace_color,  # Set border color to match
                    borderwidth=2,
                    borderpad=1,  # Add some padding for rounded effect
                )

            return fig

    @staticmethod
    def get_top_categories(category_totals, show_count: int = 5, target_coverage: float = 0.8, max_count: int = 10):
        """get the top categories, if show_count is not None, returns the N to categories where N = show_count.
        otherwise, calcualte N such that the N categories represent cover target_coverage percent of the total"""
        if show_count is None:
            # Calculate total sum of all categories
            total_sum = category_totals.sum()

            # Calculate cumulative sum and find the index for top categories that represent more than 80% of the total
            category_totals_sorted = category_totals.sort_values(ascending=False)
            cumulative_sum = category_totals_sorted.cumsum()

            cum_coverage = cumulative_sum[cumulative_sum <= total_sum * target_coverage]
            if len(cum_coverage) > max_count:
                top_n = cum_coverage.index[max_count - 1]
            else:
                top_n = cum_coverage.index[-1]

            # Get the top categories
            top_categories = category_totals_sorted.loc[:top_n].index
        else:
            top_categories = category_totals.nlargest(show_count).index
        return top_categories

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
        incremental_values = [
            f"{x / 1000:,.0f}K" if abs(x) > 1000 else f"{x:,.0f}"
            for x in list(categorised_flows.values) + [-categorised_flows.values.sum()]
        ]

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
                customdata=incremental_values,
                hovertemplate="Category: %{x}<br>Value: %{customdata}<extra></extra>",
            )
        )

        title = f"Income/Spending Summary {date_range[1].year}" if date_range is not None else "Income/Spending Summary"
        fig.update_layout(showlegend=False, margin=dict(t=50), title=title)

        return fig

    def get_asset_plot(self, df, asset):
        """plot asset information: Quantity, Unit Price, Total Value"""

        fig = make_subplots(rows=1, cols=2, specs=[[{"secondary_y": True}, {"secondary_y": True}]], horizontal_spacing=0.15)

        if asset not in df.index:
            return fig

        fig.append_trace(go.Scatter(x=df.loc[asset].index, y=df.loc[asset].Quantity_cumsum, name="Asset Quantity"), row=1, col=1)

        fig.add_trace(
            go.Scatter(x=df.loc[asset].index, y=df.loc[asset].AssetPriceInRefCurrency, name="Asset Price"), secondary_y=True, row=1, col=1
        )

        fig.append_trace(go.Scatter(x=df.loc[asset].index, y=df.loc[asset].Value, name="Total Asset Value"), row=1, col=2)

        fig.update_layout(
            title=f"{asset} Investment Evolution",
            margin=dict(t=50, b=20, r=0, l=0, pad=4),
            yaxis=dict(
                title=dict(text="Quantity"),
            ),
            yaxis2=dict(
                title=dict(text="Unit Price (USD)"),
            ),
            yaxis3=dict(
                title=dict(text="Price (USD)"),
            ),
            legend={"xanchor": "right", "x": 1.08},
        )

        return fig

    def get_capital_gain_brkdn(self, date_range, target_coverage: float = 0.95):
        """Get a Breakdown of the Capital Gain"""
        df = self.transformation_manager.get_values_by_asset(date_range)
        df_capital_gain = df.groupby("AssetMapped").agg(
            {"CapitalGain": "sum", "AssetPriceInRefCurrency": ["first", "last"], "Value": "first"}
        )

        df_cap_excl_ccy = df_capital_gain[[not is_ccy(idx) for idx in df_capital_gain.index]]

        df_cap_excl_ccy.columns = [" ".join(col).strip() for col in df_cap_excl_ccy.columns.values]
        df_cap_excl_ccy = df_cap_excl_ccy.rename(
            columns={
                "CapitalGain sum": "CapitalGain",
                "AssetPriceInRefCurrency first": "StartPrice",
                "AssetPriceInRefCurrency last": "EndPrice",
                "Value first": "StartValue",
            }
        )

        df_cap_excl_ccy["AbsCapitalGain"] = df_cap_excl_ccy["CapitalGain"].abs()
        df_cap_excl_ccy = df_cap_excl_ccy.sort_values(by="AbsCapitalGain", ascending=False)
        df_cap_excl_ccy = df_cap_excl_ccy.reset_index()

        cumulative_sum = df_cap_excl_ccy["AbsCapitalGain"].cumsum()
        total_sum = df_cap_excl_ccy["AbsCapitalGain"].sum()
        cum_coverage = cumulative_sum[cumulative_sum <= total_sum * target_coverage]

        df_cap_excl_ccy["Return"] = (df_cap_excl_ccy["EndPrice"] - df_cap_excl_ccy["StartPrice"]) / df_cap_excl_ccy["StartPrice"]

        df_out = df_cap_excl_ccy.iloc[: cum_coverage.index[-1]].set_index("AssetMapped")[["StartValue", "CapitalGain", "Return"]]
        df_total = pd.DataFrame(
            columns=["AssetMapped", "StartValue", "CapitalGain", "Return"],
            data=[
                [
                    "Total",
                    df_cap_excl_ccy.StartValue.sum(),
                    df_cap_excl_ccy.CapitalGain.sum(),
                    df_cap_excl_ccy.CapitalGain.sum() / df_cap_excl_ccy.StartValue.sum(),
                ]
            ],
        ).set_index("AssetMapped")

        if len(df_out.index) == 0:
            return df_total, None
        return pd.concat([df_out, df_total]), self.get_asset_plot(df, df_out.index[0])
