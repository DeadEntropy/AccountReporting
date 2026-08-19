# AccountReporting (`bkanalysis`)

A Python package to aggregate personal bank, brokerage, pension and crypto account data from heterogeneous CSV exports, harmonize it into a single transaction format, enrich it with market prices, and produce statistics and interactive plots to monitor a personal financial situation.

It is the engine behind [AccountReportingWebapp](https://github.com/DeadEntropy/AccountReportingWebapp), a Dash dashboard, and can also be used directly from notebooks (see `notebooks/`).

## Installation

```bash
pip install git+https://github.com/DeadEntropy/AccountReporting
```

## Configuration

The package is driven by a configuration file which specifies where to load the data from, the expected columns of each institution's export, the mapping files used to categorise transactions, and the market data sources. By default the package looks for the configuration in `config/config.ini` (relative to the working directory).

Key sections:

- `[IO]` — input folder (`folder_lake`) and output paths.
- `[Mapping]` — paths to the memo/type/subtype/master-type mapping CSVs used to categorise transactions.
- `[Market]` — instrument-to-source overrides (Yahoo Finance, file, hardcoded, Nutmeg) and instruments to preload.
- One section per institution (e.g. `[Barclays]`, `[Revolut]`, `[Citi]`, `[Fidelity]`...) describing the expected columns and account metadata. Around 25 institution formats are supported (see `bkanalysis/transforms/account_transforms/`).

## How the pipeline works

1. **Load** — `transforms.master_transform.Loader` scans the data lake folder and dispatches each file to the institution-specific transform that recognises its columns; everything is normalised to a common schema (`Date, Account, Amount, Subcategory, Memo, Currency, AccountType, SourceFile`).
2. **Process** — `process.process.Process` cleans the memos, maps them to categories (Type / SubType / MasterType plus their "Full" display names), applies manual overrides, and pairs up offsetting intra-account transfers (`process.iat_identification`).
3. **Price** — `managers.MarketManager` resolves each asset to a market symbol, loads price histories (Yahoo Finance via `yfinance`, files, or hardcoded values) and converts everything into a reference currency.
4. **Analyse** — `managers.TransformationManager` and `managers.FigureManager` compute values by asset, flows, category breakdowns, capital gains and saving rates, and build the Plotly figures used by the webapp.

## High-level use (managers API)

```python
from datetime import datetime
from bkanalysis.managers import DataManager, MarketManager, TransformationManager, FigureManager

data_manager = DataManager()
data_manager.load_data_from_disk()          # parse the raw files from the data lake

market_manager = MarketManager("USD")
market_manager.load_prices(data_manager)    # download/load market prices

transformation_manager = TransformationManager(data_manager, market_manager)
figure_manager = FigureManager(transformation_manager)

fig = figure_manager.get_figure_timeseries(date_range=[datetime(2025, 1, 1), datetime(2025, 12, 31)])
fig.show()
```

Both `DataManager` and `MarketManager` support `to_disk()` / `load_pregenerated_data()`, so the expensive parsing and market-data steps can be done once (e.g. in `notebooks/run_data_manager.ipynb`) and the results served to the webapp as plain CSVs.

## Modules

| Module | Purpose |
|---|---|
| `transforms` | Per-institution CSV loaders and the master loader that dispatches files. |
| `process` | Memo cleaning, category mapping, overrides, intra-account-transfer identification. |
| `managers` | Modern API: `DataManager`, `MarketManager`, `TransformationManager` (+ cache variant), `FigureManager`. |
| `market` | Market data loading (Yahoo Finance, files, hardcoded, Nutmeg) and FX conversion. |
| `salary` | Salary tracking across multiple payrolls, including outstanding/carried-over salary. |
| `projection` | Wealth projection with growth/volatility scenarios and uncertainty bands. |
| `portfolio` | Portfolio time series and benchmark comparison. |
| `tax` | UK tax and Nutmeg-specific capital-gain helpers. |
| `ui` | Legacy plotting API kept for the notebooks (`ui.py`, `ui_old.py`). New code should use `managers`. |

## Tests

```bash
pytest tests/
pytest --cov=bkanalysis --cov-report=html tests/
```

See [TESTING.md](TESTING.md) for the test layout, fixtures and coverage status. External market-data calls are fully mocked in the test suite.
