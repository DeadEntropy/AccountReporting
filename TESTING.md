# Testing Guide & Progress Report for AccountReporting

**Overall Coverage: 45.00%** (387 tests passing, 1 skipped)
**Target: 75%**
**Improvement: +5.53% from Phase 2**

---

## Quick Start

### Running Tests
```bash
# Run all tests
pytest tests/

# Run tests with coverage report
pytest --cov=bkanalysis --cov-report=term-missing tests/

# Run tests with HTML coverage report
pytest --cov=bkanalysis --cov-report=html tests/
# Open htmlcov/index.html in a browser to view detailed report

# Run specific test file
pytest tests/unit/test_tax.py -v

# Run specific test class
pytest tests/unit/test_tax.py::TestTax -v

# Run specific test
pytest tests/unit/test_tax.py::TestTax::test_tax_basic_band -v
```

### Coverage Threshold
The project enforces a **75% code coverage** threshold. CI/CD will fail if coverage drops below this level.

---

## Test Structure

Tests follow pytest conventions:
- Test files: `tests/unit/test_*.py`
- Test classes: `class Test*:`
- Test methods: `def test_*():`

Example:
```python
def test_example_feature(self):
    """Test description."""
    assert result == expected_value
```

### Fixtures

Reusable test fixtures are defined in `tests/conftest.py`:
- `config`: ConfigParser object with test configuration
- `config_path`: Path to test config file
- `master_transform_loader`: MasterTransform Loader instance
- `iat_identifier`: IatIdentification instance
- `test_data_path`: Path to test data directory

Example usage:
```python
def test_my_feature(self, config):
    """Test using config fixture."""
    dm = DataManager(config)
    assert dm.config is not None
```

---

## Current Testing Progress

### Phase 1: Core Transform & Market Module Testing (Complete ✅)

#### Summary
- Phase 1.1: 4 of 31 account transforms tested (13% coverage)
- Phase 1.2: Manager classes fully tested (40 tests, high coverage)
- Phase 1.3: Market module with complete yfinance mocking (40 tests)

#### Phase 1.1: Account Transforms (4/31 completed)
**Status:** In Progress
**Coverage:** ~13% of account transforms module
**Tests:** 74 tests across 11 account transform classes

**Completed Transforms:**
- ✅ Barclays (14 tests, 100%)
- ✅ Chase (19 tests, 94.12%)
- ✅ Chase Business (16 tests, 94.12%)
- ✅ Coinbase (10 tests, 100%)
- ✅ Coinbase Pro (12 tests, 100%)
- ✅ Clone (13 tests, 100%)
- ✅ Discover (18 tests, 100%)
- ✅ First Republic (19 tests, 100%)
- ✅ Revolut (10 tests, 94.2%)
- ⚠️ BNP (16 tests, 87.1% - Cash transform complete, Stock transform at 84.85%)
- ⚠️ Capital One (1 test, 92.86%)

**Remaining Transforms (27/31):** Citi, Fidelity, Lloyds, Marcus, Mortgage Script, Nutmeg, UBS variants, Vault, and others

#### Phase 1.2: Manager Classes (Complete ✅)
**Status:** Complete
**Coverage:** 84.21% (data_manager), 79.17% (transformation_manager_cache), 73.10% (transformation_manager), 91.30% (manager_helper)
**Tests:** 40 tests

**Modules:**
- ✅ data_manager.py (84.21%)
- ✅ transformation_manager.py (73.10%)
- ✅ transformation_manager_cache.py (79.17%)
- ✅ manager_helper.py (91.30%)

#### Phase 1.3: Market Module (Complete ✅)
**Status:** Complete
**Coverage:** 52.62% (market module overall)
**Tests:** 40 comprehensive tests with complete yfinance mocking

**Modules:**
- ✅ price.py (100% - 4 statements)
- ✅ market.py (54.00% - 28 lines missed, core functionality covered)
- ✅ market_loader.py (57.53% - 35 lines missed, multi-source loading covered)
- ⚠️ market_prices.py (47.88% - 59 lines missed, cache/ISIN functions need coverage)

**Test Categories:**
- TestPrice: 3 tests (initialization scenarios)
- TestMarket: 11 tests (price lookup, interpolation, currency conversion)
- TestMarketPrices: 10 tests (symbol resolution, currency mapping, yfinance mocking)
- TestMarketLoader: 12 tests (multi-source data loading: Yahoo, hardcoded, file, Nutmeg)
- TestMarketIntegration: 2 tests (complete workflows)

**Key Implementation:** Complete mocking of yfinance dependency - zero API calls in tests

---

### Phase 2: Supporting Modules (Complete ✅)

#### Phase 2.1: Portfolio Module (Complete ✅)
**Status:** Complete
**Coverage Achieved:**
- cache.py: 92.31% (up from 0%)
- portfolio.py: 59.77% (up from 0%)
**Tests:** 18 tests

**Key Features Tested:**
- CacheDict initialization and JSON persistence (with mocking)
- get_benchmarks() - benchmark time series normalization
- process_stock() - stock data processing with market_prices integration
- clean_time_series() - time series data cleaning
- get_portfolio_ts() - portfolio aggregation and normalization
- total_return() - return calculations
**Test Patterns:** Complete mocking of market_prices external calls

#### Phase 2.2: Process Module (Complete ✅)
**Status:** Complete
**Coverage Achieved:**
- process.py: 46.15% (up from 13.46%)
- process_helper.py: 31.11% (up from 8.89%) 
- iat_identification.py: 84.81% (up from 56.96%)
- status.py: 58.33% (up from 29.17%)
**Tests:** 28 tests

**Process Helper Functions Tested:**
- get_adjusted_month() / get_adjusted_year() - fiscal date adjustments
- get_fiscal_year() - fiscal year calculation
- get_year_to_date() - age in years calculation

**IatIdentification Tests:**
- map_iat() - intra-account transfer identification with date windows
- map_iat_fx() - FX transfer identification

**Process Class Tests:**
- memo mapping (simple & case-insensitive)
- type/subtype extraction and mapping
- memo cleaning (Amazon variants, asterisk removal, space normalization)
- get_full_type() - type name lookups
- LastUpdate date formatting

**Test Patterns:** Fixture-based config setup, DataFrame mocking

#### Phase 2.3: Projection Module (Complete ✅)
**Status:** Complete
**Coverage Achieved:** projection.py: 64.91% (up from 14.04%)
**Tests:** 40 tests (All passing ✅)

**Projection Functions Tested:**
- thousands() - number formatter for K/M notation
- project() - recursive wealth projection with growth, volatility, contributions
- project_full() - full projection with uncertainty bounds
- Realistic, conservative, and aggressive scenarios
- Multi-year projections (up to 50 years)

**Test Coverage:**
- Growth rate scenarios (positive, negative, zero)
- Volatility impact on projections
- Contribution scenarios (single/multiple years)
- Uncertainty bounds ordering (extreme_low < low < mean < up < extreme_up)
- Edge cases (zero initial, very large timeframes, negative growth)

**Key Achievement:** All 40 projection tests passing with diverse scenarios

---

## Test Execution Summary
```
Total Tests: 387 passed, 1 skipped (81 new Phase 2 tests)
Execution Time: 3.43 seconds
Environment: Python 3.11.9, pytest-9.0.2, pytest-cov-7.0.0
Phase 2 Test Files: test_portfolio.py (18), test_process.py (28), test_projection.py (40)
```

## Coverage Improvement Summary
| Phase | Tests | Coverage | Notes |
|-------|-------|----------|-------|
| Phase 1.1-1.3 | 307 | 39.47% | Account transforms, managers, market module |
| Phase 2 | 81 | +5.53% | Portfolio, process, projection modules |
| **Total** | **387** | **45.00%** | On track to 75% target |

---

## Coverage by Category

### Excellent (80%+)
- barclays_transform.py: 100%
- chase_transform.py: 94.12%
- chasebusiness_transform.py: 94.12%
- coinbase_pro_transform.py: 100%
- coinbase_transform.py: 100%
- discover_transform.py: 100%
- fidelity_transform.py: 100%
- first_republic_transform.py: 100%
- manager_helper.py: 91.30%
- data_manager.py: 84.21%
- iat_identification.py: 84.81%
- portfolio/cache.py: 92.31%

### Good (50-79%)
- market_loader.py: 57.53%
- market.py: 54.00%
- transformation_manager.py: 73.10%
- transformation_manager_cache.py: 79.17%
- tax.py: 61.22%
- projection.py: 64.91%
- status.py: 58.33%
- portfolio.py: 59.77%

### Needs Work (<50%)
- market_prices.py: 47.88%
- process.py: 46.15%
- process_helper.py: 31.11%
- salary.py: 15.05%
- tax/nutmeg.py: 6.04%
- UI modules: 0-30%

---

## Writing New Tests

### Best Practices

1. **Test one thing** - Each test should verify a single behavior
2. **Use descriptive names** - Test name should describe what is being tested
3. **Use fixtures** - Leverage existing fixtures from `conftest.py`
4. **Mock external dependencies** - Use `unittest.mock` for external calls (especially yfinance)
5. **Write docstrings** - Document what each test verifies

### Example Test

```python
"""Tests for feature module."""
import pytest
from bkanalysis.feature import MyClass

class TestMyClass:
    """Tests for MyClass."""

    def test_initialization(self, config):
        """Verify MyClass initializes correctly."""
        obj = MyClass(config)
        assert obj is not None
        assert obj.config is not None

    def test_method_returns_correct_value(self, config):
        """Verify method returns expected value."""
        obj = MyClass(config)
        result = obj.my_method()
        assert result == "expected_value"

    @pytest.fixture
    def sample_data(self):
        """Provide sample data for tests."""
        return {"key": "value"}

    def test_with_sample_data(self, sample_data):
        """Test with provided sample data."""
        assert sample_data["key"] == "value"
```

### Mocking Example

Avoid using Mocking as much as possible. Only use Mocking to mock calls to yfinance and other external APIs.

---

## Next Steps

### Recommended: Phase 1.1 Continuation
Continue testing remaining **27/31 account transforms** (currently 4 done):
- Estimated: 100+ additional tests
- Expected coverage gain: ~15-20%
- Would bring account transforms from ~13% to 75-80%
- After completion: ~50-55% overall coverage

### Alternative: Fill Coverage Gaps in Phase 2
- market_prices.py: 47.88% → target 65%+ (cache/ISIN functions)
- portfolio.py: 59.77% → target 75%+ 
- process.py: 46.15% → target 65%+
- Estimated: 50+ additional tests

### Phase 3: UI/Advanced Modules
- figure_manager.py: 10.51% (complex visualization logic)  
- tax/nutmeg.py: 6.04% (specialized tax calculation)
- salary.py: 15.05% (salary analysis)
- Estimated: 150+ tests needed

**Recommendation:** Continue Phase 1.1 for quick coverage gains while maintaining broad testing coverage across all modules.

---

## CI/CD Integration

Tests run automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Tests run on Python 3.9, 3.10, and 3.11
- Coverage must be ≥75% to pass CI

GitHub Actions workflow: `.github/workflows/test-coverage.yml`

To view coverage:
1. Check CI job output for summary
2. Codecov artifacts available in GitHub Actions artifacts
3. Run locally: `pytest --cov=bkanalysis --cov-report=html` then open `htmlcov/index.html`

---

## Debugging Test Failures

### Run with more verbose output
```bash
pytest tests/unit/test_file.py -vv
```

### Show local variables in failures
```bash
pytest tests/unit/test_file.py -l
```

### Stop at first failure
```bash
pytest tests/unit/test_file.py -x
```

### Run only last failed tests
```bash
pytest tests/unit/test_file.py --lf
```

### Show print statements
```bash
pytest tests/unit/test_file.py -s
```

---

## Common Issues & Solutions

### Issue: Import errors in tests
**Solution:** Ensure `conftest.py` fixtures are properly imported. Tests in `tests/unit/` will automatically have access to fixtures from `tests/conftest.py`.

### Issue: Config not found
**Solution:** Tests use fixture config from `tests/conftest.py`. Ensure `tests/unit/config.ini` exists and test data paths are correct.

### Issue: Dataframe column mismatches
**Solution:** Check the actual columns required by the module. Examine the source code or error message for expected columns.

### Issue: Mocks not working
**Solution:** 
- Patch where the object is *used*, not where it's defined
- Use `@patch` decorator for class-level mocks
- Use `unittest.mock.patch()` context manager for inline mocks

---

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Pandas Testing Guide](https://pandas.pydata.org/docs/reference/testing.html)

---

## Contributing Tests

When adding new tests:
1. Place in appropriate `tests/unit/test_*.py` file
2. Follow pytest naming conventions
3. Use existing fixtures from `conftest.py`
4. Add docstrings to explain what is being tested
5. Run `pytest --cov=bkanalysis` to verify coverage impact
6. Ensure CI/CD passes before merging

For questions about testing patterns, refer to existing test files as examples.
