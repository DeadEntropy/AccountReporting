"""Tests for market module (market.py, market_loader.py, market_prices.py)."""
import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import pandas as pd
import datetime as dt
from datetime import datetime
import configparser
import yfinance as yf

from bkanalysis.market.market import Market
from bkanalysis.market.price import Price
from bkanalysis.market.market_loader import MarketLoader, SOURCE
from bkanalysis.market import market_prices as mp


@pytest.fixture
def market_config():
    """Create a valid config with Market section for testing."""
    config = configparser.ConfigParser()
    config['Market'] = {
        'source_map': "{'AAPL': ['YAHOO'], 'TEST': ['HARDCODED', {'01/01/2023': 100.0}, 'USD']}"
    }
    return config


class TestPrice:
    """Tests for Price data class."""

    def test_init_with_value_and_currency(self):
        """Test Price initialization."""
        price = Price(100.5, "GBP")
        assert price.value == 100.5
        assert price.currency == "GBP"

    def test_init_with_zero_value(self):
        """Test Price with zero value."""
        price = Price(0.0, "USD")
        assert price.value == 0.0
        assert price.currency == "USD"

    def test_init_with_different_currencies(self):
        """Test Price with various currency codes."""
        for ccy in ["GBP", "USD", "EUR", "JPY", "CHF"]:
            price = Price(50.0, ccy)
            assert price.currency == ccy


class TestMarket:
    """Tests for Market class."""

    @staticmethod
    def _create_sample_prices():
        """Helper to create sample price data."""
        dates = pd.date_range('2023-01-01', periods=5)
        return {
            'AAPL': {
                date: Price(100.0 + i, 'USD')
                for i, date in enumerate(dates)
            },
            'GBPUSD=X': {
                date: Price(1.2 + i * 0.01, 'USD')
                for i, date in enumerate(dates)
            },
        }

    def test_init_with_valid_prices(self):
        """Test Market initialization with valid price data."""
        prices = self._create_sample_prices()
        market = Market(prices)
        
        assert market.supported_instr() == ['AAPL', 'GBPUSD=X']
        assert market.linear_interpolation == False

    def test_init_with_linear_interpolation_enabled(self):
        """Test Market initialization with linear interpolation."""
        prices = self._create_sample_prices()
        market = Market(prices, linear_interpolation=True)
        
        assert market.linear_interpolation == True

    def test_supported_instr_returns_all_instruments(self):
        """Test that supported_instr returns all available instruments."""
        prices = self._create_sample_prices()
        market = Market(prices)
        
        instruments = market.supported_instr()
        assert len(instruments) == 2
        assert 'AAPL' in instruments
        assert 'GBPUSD=X' in instruments

    def test_get_price_with_exact_date(self):
        """Test get_price with exact date match."""
        prices = self._create_sample_prices()
        market = Market(prices)
        
        date = pd.Timestamp('2023-01-01')
        price = market.get_price('AAPL', date)
        
        assert price.value == 100.0
        assert price.currency == 'USD'

    def test_get_price_with_interpolation_disabled(self):
        """Test get_price uses closest date when exact date not found."""
        prices = self._create_sample_prices()
        market = Market(prices, linear_interpolation=False)
        
        # Request price for date not in data
        date_not_in_data = pd.Timestamp('2023-01-01 12:00:00')
        price = market.get_price('AAPL', date_not_in_data)
        
        assert isinstance(price, Price)
        assert price.currency == 'USD'

    def test_get_price_with_interpolation_enabled(self):
        """Test get_price with linear interpolation."""
        dates = [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02')]
        prices = {
            'AAPL': {
                dates[0]: Price(100.0, 'USD'),
                dates[1]: Price(110.0, 'USD'),
            }
        }
        market = Market(prices, linear_interpolation=True)
        
        # Request exact date - should return exact price
        price = market.get_price('AAPL', dates[0])
        
        assert isinstance(price, Price)
        assert price.value == 100.0

    def test_get_price_instrument_not_found(self):
        """Test get_price raises error for missing instrument."""
        prices = self._create_sample_prices()
        market = Market(prices)
        
        with pytest.raises(Exception):
            market.get_price('NONEXISTENT', pd.Timestamp('2023-01-01'))

    def test_get_price_in_currency_same_currency(self):
        """Test get_price_in_currency when price is already in target currency."""
        prices = self._create_sample_prices()
        market = Market(prices)
        
        date = pd.Timestamp('2023-01-01')
        price = market.get_price_in_currency('AAPL', date, 'USD')
        
        assert price == 100.0
        assert isinstance(price, (float, int))

    def test_get_price_in_currency_different_currency(self):
        """Test get_price_in_currency with currency conversion."""
        dates = pd.date_range('2023-01-01', periods=3)
        prices = {
            'AAPL': {
                date: Price(100.0, 'USD')
                for date in dates
            },
            'USDGBP=X': {
                date: Price(0.8, 'GBP')
                for date in dates
            },
        }
        market = Market(prices)
        
        date = pd.Timestamp('2023-01-01')
        price = market.get_price_in_currency('AAPL', date, 'GBP')
        
        # Returns float value, not Price object
        assert isinstance(price, (float, int))
        assert price == 80.0  # 100 * 0.8

    def test_get_price_in_currency_missing_instrument(self):
        """Test get_price_in_currency raises error for missing instrument."""
        prices = self._create_sample_prices()
        market = Market(prices)
        
        with pytest.raises(Exception):
            market.get_price_in_currency('MISSING', pd.Timestamp('2023-01-01'), 'USD')


class TestMarketPrices:
    """Tests for market_prices module with mocked yfinance."""

    @patch('bkanalysis.market.market_prices.yf.Ticker')
    def test_get_symbol_with_currency(self, mock_ticker):
        """Test get_symbol with currency code."""
        result = mp.get_symbol('USD', 'GBP')
        assert result == 'USDGBP=X'

    @patch('bkanalysis.market.market_prices.yf.Ticker')
    def test_get_symbol_with_crypto(self, mock_ticker):
        """Test get_symbol with cryptocurrency."""
        result = mp.get_symbol('BTC', 'USD')
        assert result == 'BTC-USD'

    @patch('bkanalysis.market.market_prices.yf.Ticker')
    def test_get_symbol_with_isin(self, mock_ticker):
        """Test get_symbol with ISIN code."""
        with patch('bkanalysis.market.market_prices.get_with_isin_map', return_value='AAPL'):
            result = mp.get_symbol('US0378331005', 'USD')
            assert result == 'AAPL'

    @patch('bkanalysis.market.market_prices.yf.Ticker')
    def test_get_symbol_with_ticker(self, mock_ticker):
        """Test get_symbol with ticker symbol."""
        result = mp.get_symbol('AAPL', 'USD')
        assert result == 'AAPL'

    @patch('bkanalysis.market.market_prices.yf.Ticker')
    def test_get_symbol_with_none(self, mock_ticker):
        """Test get_symbol returns None when instrument is None."""
        result = mp.get_symbol(None, 'USD')
        assert result is None

    @patch('bkanalysis.market.market_prices.yf.Ticker')
    def test_get_currency_with_currency_map(self, mock_ticker):
        """Test get_currency with pre-mapped symbol."""
        result = mp.get_currency('FHQP.F')
        assert result == 'EUR'

    @patch('bkanalysis.market.market_prices.yf.Ticker')
    def test_get_currency_from_yahoo(self, mock_ticker):
        """Test get_currency queries yfinance."""
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {'currency': 'usd'}
        mock_ticker.return_value = mock_ticker_instance
        
        result = mp.get_currency('AAPL')
        assert result == 'USD'
        mock_ticker.assert_called_once_with('AAPL')

    @patch('bkanalysis.market.market_prices.yf.Ticker')
    def test_get_currency_error_handling(self, mock_ticker):
        """Test get_currency raises exception on error."""
        mock_ticker.side_effect = Exception('API Error')
        
        with pytest.raises(Exception):
            mp.get_currency('INVALID')

    @patch('bkanalysis.market.market_prices.yf.Ticker')
    def test_get_history_mocked(self, mock_ticker):
        """Test get_history with mocked yfinance."""
        # Create mock data
        dates = pd.date_range('2023-01-01', periods=3)
        mock_history = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0]
        }, index=dates)
        
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_history
        mock_ticker.return_value = mock_ticker_instance
        
        result = mp.get_history('AAPL', '3d')
        
        assert len(result) == 3
        assert list(result['Close']) == [100.0, 101.0, 102.0]
        mock_ticker.assert_called_with('AAPL')

    @patch('bkanalysis.market.market_prices.get_history')
    def test_get_spot_price_currency_to_currency(self, mock_get_history):
        """Test get_spot_price for currency conversion."""
        dates = pd.date_range('2023-01-01', periods=1)
        # Create DataFrame with Close column and Date as regular column (not index)
        mock_history = pd.DataFrame({
            'Date': dates,
            'Close': [1.2]
        })
        
        mock_get_history.return_value = mock_history
        
        result = mp.get_spot_price('EUR', 'USD')
        
        assert result == 1.2
        mock_get_history.assert_called_with('EURUSD=X', '1y')

    @patch('bkanalysis.market.market_prices.get_history')
    def test_get_spot_price_same_currency(self, mock_get_history):
        """Test get_spot_price returns 1.0 for same currency."""
        result = mp.get_spot_price('USD', 'USD')
        assert result == 1.0
        mock_get_history.assert_not_called()

    @patch('bkanalysis.market.market_prices.get_history')
    def test_get_spot_price_none_instrument(self, mock_get_history):
        """Test get_spot_price with None instrument."""
        result = mp.get_spot_price(None, 'USD')
        assert result is None
        mock_get_history.assert_not_called()

    @patch('bkanalysis.market.market_prices.get_history')
    @patch('bkanalysis.market.market_prices.get_spot_price')
    def test_get_spot_prices_multiple(self, mock_spot_price, mock_get_history):
        """Test get_spot_prices with multiple instruments."""
        mock_spot_price.side_effect = [1.2, 1.1]
        
        result = mp.get_spot_prices(['EUR', 'GBP'], 'USD')
        
        assert result['EUR'] == 1.2
        assert result['GBP'] == 1.1
        assert mock_spot_price.call_count == 2


class TestMarketLoader:
    """Tests for MarketLoader class with mocked dependencies."""

    def test_init_with_config(self, market_config):
        """Test MarketLoader initialization with config."""
        loader = MarketLoader(market_config)
        
        assert loader.config is not None
        assert loader.source_default == SOURCE.YAHOO

    def test_init_without_config_raises_error(self):
        """Test MarketLoader initialization without valid config raises error."""
        with patch('bkanalysis.config.config_helper.source', '/nonexistent/path'):
            with pytest.raises(OSError):
                MarketLoader(None)

    def test_init_missing_market_section(self):
        """Test MarketLoader raises error when Market section missing."""
        # Create minimal config without Market section
        minimal_config = configparser.ConfigParser()
        
        with pytest.raises(Exception):
            MarketLoader(minimal_config)

    @patch('bkanalysis.market.market_prices.get_history')
    def test_get_history_from_yahoo(self, mock_get_history):
        """Test get_history_from_yahoo with mocked yfinance."""
        dates = pd.date_range('2023-01-01', periods=2)
        mock_history = pd.DataFrame({
            'Close': [100.0, 101.0]
        }, index=dates)
        mock_history.index.name = 'Date'
        
        mock_get_history.return_value = mock_history
        
        result = MarketLoader.get_history_from_yahoo('AAPL', '2d')
        
        assert len(result) == 2
        assert result[dates[0]].value == 100.0
        assert result[dates[1]].value == 101.0

    def test_get_history_from_hardcoded(self):
        """Test get_history_from_hardcoded."""
        values = {
            '01/01/2023': 100.0,
            '02/01/2023': 101.0,
        }
        
        result = MarketLoader.get_history_from_hardcoded(values, 'USD')
        
        assert len(result) == 2
        assert result[datetime(2023, 1, 1)].value == 100.0
        assert result[datetime(2023, 1, 2)].value == 101.0

    def test_get_history_from_file(self, tmp_path):
        """Test get_history_from_file."""
        # Create test CSV file
        csv_path = tmp_path / "prices.csv"
        csv_content = """Unit Price Date,Unit Price,Currency code
01/01/2023,100.0,USD
02/01/2023,101.0,USD
"""
        csv_path.write_text(csv_content)
        
        result = MarketLoader.get_history_from_file(str(csv_path))
        
        assert len(result) == 2
        assert result[datetime(2023, 1, 1)].value == 100.0
        assert result[datetime(2023, 1, 2)].value == 101.0
        assert result[datetime(2023, 1, 1)].currency == 'USD'

    @patch('bkanalysis.market.market_loader.mp.get_currency')
    def test_get_currency(self, mock_get_currency):
        """Test get_currency static method."""
        mock_get_currency.return_value = 'USD'
        
        result = MarketLoader.get_currency('AAPL')
        
        assert result == 'USD'
        mock_get_currency.assert_called_once_with('AAPL')

    @patch('bkanalysis.market.market_loader.mp.get_symbol')
    def test_get_symbol(self, mock_get_symbol):
        """Test get_symbol static method."""
        mock_get_symbol.return_value = 'AAPL'
        
        result = MarketLoader.get_symbol('AAPL', 'USD')
        
        assert result == 'AAPL'
        mock_get_symbol.assert_called_once_with('AAPL', 'USD')

    @patch('bkanalysis.market.market_prices.get_history')
    @patch('bkanalysis.market.market_prices.get_currency')
    def test_load_with_yahoo_source(self, mock_get_currency, mock_get_history, market_config):
        """Test load method with YAHOO source."""
        # Mock the market_prices functions
        dates = pd.date_range('2023-01-01', periods=2)
        mock_history = pd.DataFrame({
            'Close': [100.0, 101.0]
        }, index=dates)
        mock_history.index.name = 'Date'
        
        mock_get_history.return_value = mock_history
        mock_get_currency.return_value = 'USD'
        
        loader = MarketLoader(market_config)
        
        result = loader.load(['AAPL'], 'USD', '2d')
        
        assert isinstance(result, dict)
        assert len(result) > 0

    @patch('bkanalysis.market.market_prices.yf.Ticker')
    def test_load_with_empty_instruments(self, mock_ticker, market_config):
        """Test load with empty instruments list."""
        loader = MarketLoader(market_config)
        
        result = loader.load([], 'USD', '1d')
        
        # Should return empty or minimal data
        assert isinstance(result, dict)

    @patch('bkanalysis.market.market_prices.get_history')
    def test_get_history_with_period_as_dict(self, mock_get_history, market_config):
        """Test load with period as dictionary."""
        dates = pd.date_range('2023-01-01', periods=2)
        mock_history = pd.DataFrame({
            'Close': [100.0, 101.0]
        }, index=dates)
        mock_history.index.name = 'Date'
        mock_get_history.return_value = mock_history
        
        loader = MarketLoader(market_config)
        period_dict = {'AAPL': '1d', 'GBPUSD': '1d'}
        
        # This should not raise an error
        result = loader.load(['AAPL'], 'USD', period_dict)
        assert isinstance(result, dict)

    def test_source_enum_values(self):
        """Test SOURCE enum has expected values."""
        assert SOURCE.YAHOO.value == 1
        assert SOURCE.FILE.value == 2
        assert SOURCE.HARDCODED.value == 3
        assert SOURCE.NUTMEG.value == 4


class TestMarketIntegration:
    """Integration tests for Market module."""

    @patch('bkanalysis.market.market_prices.get_history')
    @patch('bkanalysis.market.market_prices.get_currency')
    def test_full_market_workflow(self, mock_get_currency, mock_get_history, config):
        """Test complete workflow: load data and query prices."""
        # Mock yfinance responses
        dates = pd.date_range('2023-01-01', periods=3)
        mock_history = pd.DataFrame({
            'Close': [100.0, 101.0, 102.0]
        }, index=dates)
        mock_history.index.name = 'Date'
        
        mock_get_history.return_value = mock_history
        mock_get_currency.return_value = 'USD'
        
        # Create market data
        prices_data = {
            'AAPL': {
                date: Price(close, 'USD')
                for date, close in zip(dates, [100.0, 101.0, 102.0])
            }
        }
        
        # Create market and query
        market = Market(prices_data)
        
        # Query existing date
        price = market.get_price('AAPL', dates[0])
        assert price.value == 100.0
        
        # Query instrument list
        instruments = market.supported_instr()
        assert 'AAPL' in instruments

    def test_market_with_multiple_currencies(self):
        """Test Market with multiple currencies."""
        dates = pd.date_range('2023-01-01', periods=2)
        prices = {
            'AAPL': {
                dates[0]: Price(100.0, 'USD'),
                dates[1]: Price(101.0, 'USD'),
            },
            'USDGBP=X': {
                dates[0]: Price(0.8, 'GBP'),
                dates[1]: Price(0.81, 'GBP'),
            },
        }
        
        market = Market(prices)
        
        # Verify both instruments are available
        assert len(market.supported_instr()) == 2
        
        # Get price in original currency
        price_usd = market.get_price('AAPL', dates[0])
        assert price_usd.currency == 'USD'
