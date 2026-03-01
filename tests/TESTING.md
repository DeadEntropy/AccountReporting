# Testing Guide for AccountReporting

This guide provides information on running tests, checking coverage, and adding new tests for the AccountReporting package.

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

## Fixtures

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

## Current Coverage Status

**Baseline (before improvements):** 17.47%
**Current:** 36% (UI module tested)
**Target:** 75.00%

### Well-Tested Modules
- ✅ `bkanalysis/tax/tax.py` - 61.22% coverage
- ✅ `bkanalysis/transforms/master_transform.py` - 32.77% coverage
- ✅ `bkanalysis/process/iat_identification.py` - 56.96% coverage
- ✅ `bkanalysis/managers/data_manager.py` - 80.70% coverage (newly added tests)
- ✅ `bkanalysis/managers/market_manager.py` - 35.62% coverage (newly added tests)
- ✅ `bkanalysis/managers/transformation_manager.py` - 18.13% coverage (newly added tests)

### Modules Needing More Tests (0% coverage)
- `bkanalysis/portfolio/` - 0% coverage
- `bkanalysis/managers/figure_manager.py` - 10.51% coverage
- `bkanalysis/ui/` - 36% coverage (new suite added)
- Most account transform modules - 15-30% coverage
- `bkanalysis/projection/` - 0% coverage
- `bkanalysis/salary.py` - 15.05% coverage

## Writing New Tests

### Best Practices

1. **Test one thing** - Each test should verify a single behavior
2. **Use descriptive names** - Test name should describe what is being tested
3. **Use fixtures** - Leverage existing fixtures from `conftest.py`
4. **Mock external dependencies** - Use `unittest.mock` for external calls
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

```python
from unittest.mock import Mock, patch

def test_with_mock(self):
    """Test using mocks for external dependencies."""
    
    # Create a mock object
    mock_loader = Mock()
    mock_loader.load.return_value = {"asset": "price"}
    
    # Use patch decorator
    with patch('bkanalysis.module.SomeClass', mock_loader):
        result = some_function()
        assert result is not None
        mock_loader.load.assert_called_once()
```

### Testing Pandas DataFrames

```python
import pandas as pd

def test_dataframe_operation(self):
    """Test operations on DataFrames."""
    df = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=3),
        'Amount': [100, 200, 150],
        'Account': ['Acc1', 'Acc2', 'Acc1']
    })
    
    # Test grouping
    grouped = df.groupby('Account')['Amount'].sum()
    assert grouped['Acc1'] == 250
    assert grouped['Acc2'] == 200
```

## Priority Areas for Coverage Improvement

To reach 75% coverage most efficiently, focus on these areas (in priority order):

### Phase 1: High-Impact (Current: 21.64% → ~40%)
1. **Account Transforms** (31 classes, highest volume)
   - Lowest hanging fruit: Well-structured, similar patterns
   - Add basic initialization and transformation tests for each
   - Example: `tests/unit/test_account_transforms.py`

2. **Manager Classes** (Already started - continue)
   - `transformation_manager.py` - Additional methods
   - `manager_helper.py` - Utility functions
   - `transformation_manager_cache.py` - Cache functionality

3. **Market Module** (Incomplete)
   - `market.py` - Price lookup and interpolation
   - `market_loader.py` - Asset loading
   - `market_prices.py` - Price data handling

### Phase 2: Supporting Modules (40% → 60%)
1. **Portfolio Module**
   - `portfolio.py` - Portfolio operations
   - `cache.py` - Caching mechanisms

2. **Transformation Module**
   - `master_transform.py` - Already tested, enhance coverage
   - Helper utilities

3. **Process Module**
   - `process.py` - Transaction processing
   - `process_helper.py` - Helper functions

### Phase 3: UI & Visualization (60% → 75%+)
1. **UI Module** (Currently 36%)
   - `ui.py` - Main UI logic (30 unit tests added, further integration tests remain)
   - `salary.py` - Salary calculations
   - `charts/` - Charting functions

2. **Tax Module Enhancements**
   - `nutmeg.py` - Nutmeg-specific tax handling
   - Additional edge cases

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

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Pandas Testing Guide](https://pandas.pydata.org/docs/reference/testing.html)

## Contributing Tests

When adding new tests:
1. Place in appropriate `tests/unit/test_*.py` file
2. Follow pytest naming conventions
3. Use existing fixtures from `conftest.py`
4. Add docstrings to explain what is being tested
5. Run `pytest --cov=bkanalysis` to verify coverage impact
6. Ensure CI/CD passes before merging

For questions about testing patterns, refer to existing test files as examples.
