# -*- coding: utf-8 -*-
import time
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from bkanalysis.ui import ui


_df = ui.load_csv(r'D:\NicoFolder\BankAccount\lake_result_processed.csv')
_range = pd.date_range(start=_df['Date'].min().strftime("%Y-%m-%d"),
                       end=_df['Date'].max().strftime("%Y-%m-%d"), freq='m')


def _unix_time_ms(dt):
    return int(time.mktime(dt.timetuple()))


def _unix_to_datetime(unix):
    return pd.to_datetime(unix, unit='s')


def _get_marks(date_range, nth):
    result = {}
    for i, date in enumerate(date_range):
        if i % nth == 1:
            result[_unix_time_ms(date)] = str(date.strftime('%Y-%m-%d'))

    return result

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets=external_stylesheets


app = dash.Dash(__name__)


app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='freq_drop_down',
                options=[
                    {'label': 'Daily', 'value': 'd'},
                    {'label': 'Weekly', 'value': 'w'},
                    {'label': 'Monthly', 'value': 'm'}
                ],
                value='w'
            )
        ], className="one column"),
        html.Div([
            # html.H3('Column 1'),
            dcc.Graph(id='graph_wealth_plot')
        ], className="eight columns"),
        html.Div([
            # html.H3('Column 2'),
            dcc.Graph(id='sunburst_spending_breakdown')
        ], className="three columns"),
    ], className="row"),
    dcc.RangeSlider(
        id='slider_time_range',
        min=_unix_time_ms(_range.min()),
        max=_unix_time_ms(_range.max()),
        value=[_unix_time_ms(_range.max()) - 86400*365.25, _unix_time_ms(_range.max())],
        step=86400*7,
        marks=_get_marks(_range, 8)
    ),
    html.Div(id='text_log')
])


@app.callback(
    Output('text_log', 'children'),
    [Input('slider_time_range', 'value'), Input('freq_drop_down', 'value')])
def update_output(value, freq):
    return f'You have selected "{(_unix_to_datetime(value[0]).strftime("%Y-%m-%d"))}" and {freq}'


@app.callback(
    Output('graph_wealth_plot', 'figure'),
    [Input('slider_time_range', 'value'), Input('freq_drop_down', 'value')])
def update_wealth_plot(value, freq):
    fig = ui.plot_wealth_2(_df, freq, None,
                                       [_unix_to_datetime(value[0]).strftime('%Y-%m-%d'),
                                        _unix_to_datetime(value[1]).strftime('%Y-%m-%d')])

    return fig


@app.callback(
    Output('sunburst_spending_breakdown', 'figure'),
    [Input('slider_time_range', 'value')])
def update_sunburst(value):
    fig_dum = ui.plot_sunburst(_df, ['FullType', 'FullSubType'], currency='GBP',
                               date_range=[_unix_to_datetime(value[0]).strftime('%Y-%m-%d'),
                                           _unix_to_datetime(value[1]).strftime('%Y-%m-%d')])

    fig = go.Figure(go.Sunburst(
                ids=fig_dum['data'][0]['ids'].tolist(),
                labels=fig_dum['data'][0]['labels'].tolist(),
                parents=fig_dum['data'][0]['parents'].tolist(),
                values=fig_dum['data'][0]['values'].tolist(),
                branchvalues='total'
                            )
                    )

    fig.update_layout(transition_duration=500)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
