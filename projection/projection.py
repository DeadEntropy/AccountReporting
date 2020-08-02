import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def thousands(x, pos):
    return '%1.1fM' % (x * 1e-6) if x >= 1e6 else '%1.1fK' % (x * 1e-3) if x >= 1e3 else '%1.1f' % x


def project(initial, year, growth, volatility, std_dev, contribution):
    if contribution == 0:
        return initial * np.exp(growth * year + std_dev * volatility * np.sqrt(year))
    return project(initial, year, growth, volatility, std_dev, 0) + sum([project(contribution, y, growth, volatility,
                                                                                 std_dev, 0) for y in range(0, year)])


def project_full(df_i, r, ignore_contrib=False, ignore_growth=False):
    df = df_i.copy()
    df['Forecast'] = df.apply(
        lambda row: [project(row['Amount'], i, 0.0 if ignore_growth else row['Return'],
                             0.0 if ignore_growth else row['Volatility'], 0.0,
                             0.0 if ignore_contrib else row['Contribution']) for i in r], axis=1)
    df['Forecast_Lower'] = df.apply(
        lambda row: [project(row['Amount'], i, 0.0 if ignore_growth else row['Return'],
                             0.0 if ignore_growth else row['Volatility'], -1.25,
                             0.0 if ignore_contrib else row['Contribution']) for i in r], axis=1)
    df['Forecast_Upper'] = df.apply(
        lambda row: [project(row['Amount'], i, 0.0 if ignore_growth else row['Return'],
                             0.0 if ignore_growth else row['Volatility'], 1.25,
                             0.0 if ignore_contrib else row['Contribution']) for i in r], axis=1)
    df['Forecast_Extreme_Lower'] = df.apply(
        lambda row: [project(row['Amount'], i, 0.0 if ignore_growth else row['Return'],
                             0.0 if ignore_growth else row['Volatility'], -1.96,
                             0.0 if ignore_contrib else row['Contribution']) for i in r], axis=1)
    df['Forecast_Extreme_Upper'] = df.apply(
        lambda row: [project(row['Amount'], i, 0.0 if ignore_growth else row['Return'],
                             0.0 if ignore_growth else row['Volatility'], 1.96,
                             0.0 if ignore_contrib else row['Contribution']) for i in r], axis=1)

    w = [sum(x) for x in zip(*df['Forecast'].values)]
    w_low = [sum(x) for x in zip(*df['Forecast_Lower'].values)]
    w_up = [sum(x) for x in zip(*df['Forecast_Upper'].values)]
    w_low_ex = [sum(x) for x in zip(*df['Forecast_Extreme_Lower'].values)]
    w_up_ex = [sum(x) for x in zip(*df['Forecast_Extreme_Upper'].values)]

    return w, w_low, w_up, w_low_ex, w_up_ex


def project_plot(w, w_low, w_up, w_low_ex, w_up_ex, r):
    fig, ax = plt.subplots(figsize=(12, 6))
    formatter = FuncFormatter(thousands)
    ax.yaxis.set_major_formatter(formatter)
    ax.set(xlabel='Years', ylabel='Wealth', title='Projected Wealth')

    ax.plot(r, w, '-o', label='Expected Wealth')
    ax.fill_between(r, w_low, w_up, color='b', alpha=.2, label='Less Likely')
    ax.fill_between(r, w_low_ex, w_up_ex, color='b', alpha=.1, label='Most Likely')
    plt.xticks(r, [f'{v}y' for v in r])
    ax.legend()
    ax.grid(True)
    plt.show()


def project_plot_compare(w, w_low, w_up, w_low_ex, w_up_ex, w_2, w_low_2, w_up_2, w_low_ex_2, w_up_ex_2, r,
                         ignore_fill=True):
    fig, ax = plt.subplots(figsize=(12, 6))
    formatter = FuncFormatter(thousands)
    ax.yaxis.set_major_formatter(formatter)
    ax.set(xlabel='Years', ylabel='Wealth', title='Projected Wealth')

    ax.plot(r, w, '-o', color='b', label='Expected Wealth')
    if not ignore_fill:
        ax.fill_between(r, w_low, w_up, color='b', alpha=.2, label='Less Likely')
        ax.fill_between(r, w_low_ex, w_up_ex, color='b', alpha=.1, label='Most Likely')

    ax.plot(r, w_2, '-x', color='g', label='Expected Wealth (Scenario)')
    if not ignore_fill:
        ax.fill_between(r, w_low_2, w_up_2, color='g', alpha=.2, label='Less Likely (Scenario)')
        ax.fill_between(r, w_low_ex_2, w_up_ex_2, color='g', alpha=.1, label='Most Likely (Scenario)')

    plt.xticks(r, [f'{v}y' for v in r])
    ax.legend()
    ax.grid(True)
    plt.show()
