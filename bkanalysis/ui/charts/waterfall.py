# https://stackoverflow.com/questions/77598619/create-waterfall-chart

import plotly.graph_objects as go
import pandas as pd


def plot_waterfall(data: pd.Series, title: str):
    fig = go.Figure(
        go.Waterfall(
            name="20",
            orientation="v",
            measure=["relative"] * len(data) + ["total"],
            x=list(data.index) + ["savings"],
            textposition="outside",
            text=list(data.index) + ["Savings"],
            # Plotly computes a "total" bar from the preceding relative values; the y entry is a placeholder
            y=list(data.values) + [data.values.sum()],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )

    if title is not None:
        fig.update_layout(
            title=title,
        )

    fig.update_layout(showlegend=False, margin=dict(t=50))

    return fig
