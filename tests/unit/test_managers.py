"""Tests for manager classes (data_manager, market_manager, transformation_manager)."""
import pytest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
from datetime import datetime

from bkanalysis.managers import data_manager, market_manager, transformation_manager
from bkanalysis.managers.transformation_manager_cache import TransformationManagerCache
from bkanalysis.managers import manager_helper


class TestManagerHelper:
    """Tests for manager helper utility functions."""

    def test_normalize_date_column_with_strings(self):
        """Test normalize_date_column with string date values."""
        df = pd.DataFrame({
            'Date': ['2023-01-01', '2023-02-15', '2023-03-30']
        })
        result = manager_helper.normalize_date_column(df['Date'])
        assert len(result) == 3
        assert result.iloc[0].date() == pd.Timestamp('2023-01-01').date()

    def test_normalize_date_column_with_timestamps(self):
        """Test normalize_date_column with pd.Timestamp values."""
        df = pd.DataFrame({
            'Date': pd.to_datetime(['2023-01-01', '2023-02-15', '2023-03-30'])
        })
        result = manager_helper.normalize_date_column(df['Date'])
        assert len(result) == 3
        assert result.iloc[0].date() == pd.Timestamp('2023-01-01').date()

    def test_normalize_date_column_with_mixed_formats(self):
        """Test normalize_date_column with various date formats."""
        df = pd.DataFrame({
            'Date': ['2023-01-01', '01/15/2023', '2023.03.30']
        })
        result = manager_helper.normalize_date_column(df['Date'])
        assert len(result) == 3

    def test_normalize_date_column_with_invalid_date(self):
        """Test normalize_date_column raises error for invalid date."""
        df = pd.DataFrame({
            'Date': ['2023-01-01', 'invalid-date', '2023-03-30']
        })
        with pytest.raises(ValueError):
            manager_helper.normalize_date_column(df['Date'])

    def test_is_ccy_valid_currency_pair(self):
        """Test is_ccy with valid Yahoo currency pairs."""
        assert manager_helper.is_ccy('GBPUSD=X') == True
        assert manager_helper.is_ccy('EURUSD=X') == True
        assert manager_helper.is_ccy('USDJPY=X') == True

    def test_is_ccy_invalid_currency_pair(self):
        """Test is_ccy with invalid currency pairs."""
        assert manager_helper.is_ccy('GBPUSD') == False
        assert manager_helper.is_ccy('GBP=X') == False
        assert manager_helper.is_ccy('X=X') == False

    def test_is_ccy_wrong_length(self):
        """Test is_ccy with wrong length strings."""
        assert manager_helper.is_ccy('ABC') == False
        assert manager_helper.is_ccy('ABCDEFGHIJK') == False
        assert manager_helper.is_ccy('') == False

    def test_is_ccy_lowercase(self):
        """Test is_ccy handles lowercase."""
        assert manager_helper.is_ccy('gbpusd=x') == True
        assert manager_helper.is_ccy('EurJpy=X') == True


class TestDataManager:
    """Tests for DataManager class."""

    def test_init_with_config(self, config):
        """Test DataManager initialization with config."""
        dm = data_manager.DataManager(config)
        assert dm.config is not None
        assert dm.include_xls is True
        assert dm.include_json is True
        assert dm.ignore_overrides is True
        assert dm.remove_offsetting is True
        assert dm.transactions is None

    def test_init_without_config(self):
        """Test DataManager initialization without config (uses default)."""
        with patch('bkanalysis.config.config_helper.source', 'nonexistent'):
            with pytest.raises(OSError):
                dm = data_manager.DataManager(None)

    def test_accounts_property_empty(self, config):
        """Test accounts property with no loaded transactions."""
        dm = data_manager.DataManager(config)
        # Create empty transactions DataFrame
        dm.transactions = pd.DataFrame(columns=['Account'])
        accounts = dm.accounts
        assert len(accounts) == 0

    def test_accounts_property_with_data(self, config):
        """Test accounts property with loaded transactions."""
        dm = data_manager.DataManager(config)
        # Create sample transactions
        dm.transactions = pd.DataFrame({
            'Account': ['Account1', 'Account2', 'Account1'],
            'Amount': [100, 200, 150],
            'Date': pd.date_range('2023-01-01', periods=3),
            'Asset': ['GBP', 'GBP', 'GBP'],
            'Quantity': [100, 200, 150],
            'Subcategory': ['test', 'test', 'test'],
            'Memo': ['memo1', 'memo2', 'memo3'],
            'Currency': ['GBP', 'GBP', 'GBP'],
            'MemoSimple': ['simple1', 'simple2', 'simple3'],
            'MemoMapped': ['mapped1', 'mapped2', 'mapped3'],
            'Type': ['type1', 'type2', 'type1'],
            'FullType': ['fulltype1', 'fulltype2', 'fulltype1'],
            'SubType': ['subtype1', 'subtype2', 'subtype1'],
            'FullSubType': ['fullsubtype1', 'fullsubtype2', 'fullsubtype1'],
            'MasterType': ['master1', 'master2', 'master1'],
            'FullMasterType': ['fullmaster1', 'fullmaster2', 'fullmaster1'],
            'AccountType': ['type_a', 'type_b', 'type_a'],
            'FacingAccount': ['', '', ''],
            'SourceFile': ['file1', 'file2', 'file1'],
        })
        accounts = dm.accounts
        assert list(accounts) == ['Account1', 'Account2']

    def test_assets_property_with_data(self, config):
        """Test assets property with loaded transactions."""
        dm = data_manager.DataManager(config)
        dates = pd.date_range('2023-01-01', periods=3)
        dm.transactions = pd.DataFrame({
            'Account': ['Account1', 'Account2', 'Account1'],
            'Amount': [100, 200, 150],
            'Date': dates,
            'Asset': ['GBP', 'USD', 'GBP'],
            'Quantity': [100, 200, 150],
            'Subcategory': ['test', 'test', 'test'],
            'Memo': ['memo1', 'memo2', 'memo3'],
            'Currency': ['GBP', 'USD', 'GBP'],
            'MemoSimple': ['simple1', 'simple2', 'simple3'],
            'MemoMapped': ['mapped1', 'mapped2', 'mapped3'],
            'Type': ['type1', 'type2', 'type1'],
            'FullType': ['fulltype1', 'fulltype2', 'fulltype1'],
            'SubType': ['subtype1', 'subtype2', 'subtype1'],
            'FullSubType': ['fullsubtype1', 'fullsubtype2', 'fullsubtype1'],
            'MasterType': ['master1', 'master2', 'master1'],
            'FullMasterType': ['fullmaster1', 'fullmaster2', 'fullmaster1'],
            'AccountType': ['type_a', 'type_b', 'type_a'],
            'FacingAccount': ['', '', ''],
            'SourceFile': ['file1', 'file2', 'file1'],
        })
        assets = dm.assets
        assert 'GBP' in assets
        assert 'USD' in assets
        assert assets['GBP'] == dates[0]
        assert assets['USD'] == dates[1]

    def test_to_disk(self, config, tmp_path):
        """Test saving transactions to disk."""
        dm = data_manager.DataManager(config)
        df = pd.DataFrame({
            'Date': ['2023-01-01'],
            'Account': ['test'],
            'Amount': [100],
            'Asset': ['GBP'],
            'Quantity': [100],
            'Subcategory': ['test'],
            'Memo': ['test'],
            'Currency': ['GBP'],
            'MemoSimple': ['test'],
            'MemoMapped': ['test'],
            'Type': ['test'],
            'FullType': ['test'],
            'SubType': ['test'],
            'FullSubType': ['test'],
            'MasterType': ['test'],
            'FullMasterType': ['test'],
            'AccountType': ['test'],
            'FacingAccount': ['test'],
            'SourceFile': ['test'],
        })
        dm.transactions = df

        output_path = tmp_path / "transactions.csv"
        dm.to_disk(str(output_path))

        assert output_path.exists()
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == 1
        assert loaded_df.iloc[0]['Account'] == 'test'

    def test_load_pregenerated_data(self, config, tmp_path):
        """Test loading pre-generated data from disk."""
        dm = data_manager.DataManager(config)
        
        # Create test data file
        df = pd.DataFrame({
            'Date': ['2023-01-01'],
            'Account': ['test'],
            'Amount': [100],
        })
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)

        dm.load_pregenerated_data(str(csv_path))

        assert dm.transactions is not None
        assert len(dm.transactions) == 1
        assert dm.transactions.iloc[0]['Account'] == 'test'

    def test_map_account_type(self, config):
        """Test map_account_type property."""
        dm = data_manager.DataManager(config)
        dm.transactions = pd.DataFrame({
            'Account': ['Account1', 'Account2', 'Account1'],
            'AccountType': ['type_a', 'type_b', 'type_a'],
            'Amount': [100, 200, 150],
            'Date': pd.date_range('2023-01-01', periods=3),
            'Asset': ['GBP', 'GBP', 'GBP'],
            'Quantity': [100, 200, 150],
            'Subcategory': ['test', 'test', 'test'],
            'Memo': ['memo1', 'memo2', 'memo3'],
            'Currency': ['GBP', 'GBP', 'GBP'],
            'MemoSimple': ['simple1', 'simple2', 'simple3'],
            'MemoMapped': ['mapped1', 'mapped2', 'mapped3'],
            'Type': ['type1', 'type2', 'type1'],
            'FullType': ['fulltype1', 'fulltype2', 'fulltype1'],
            'SubType': ['subtype1', 'subtype2', 'subtype1'],
            'FullSubType': ['fullsubtype1', 'fullsubtype2', 'fullsubtype1'],
            'MasterType': ['master1', 'master2', 'master1'],
            'FullMasterType': ['fullmaster1', 'fullmaster2', 'fullmaster1'],
            'FacingAccount': ['', '', ''],
            'SourceFile': ['file1', 'file2', 'file1'],
        })
        
        account_type_map = dm.map_account_type
        assert account_type_map['Account1'] == 'type_a'
        assert account_type_map['Account2'] == 'type_b'


class TestMarketManager:
    """Tests for MarketManager class."""

    def test_init(self):
        """Test MarketManager initialization."""
        mm = market_manager.MarketManager(ref_currency='GBP')
        assert mm.ref_currency == 'GBP'
        assert mm.config is None
        assert mm.prices is None
        assert mm.asset_map is None

    def test_init_with_config(self, config):
        """Test MarketManager initialization with config."""
        mm = market_manager.MarketManager(ref_currency='USD', config=config)
        assert mm.ref_currency == 'USD'
        assert mm.config is not None

    def test_to_disk(self, tmp_path):
        """Test saving prices and asset map to disk."""
        mm = market_manager.MarketManager(ref_currency='GBP')
        
        # Create test data
        mm.prices = pd.DataFrame({
            'AssetMapped': ['USD', 'EUR'],
            'Date': ['2023-01-01', '2023-01-02'],
            'Price': [1.2, 1.1],
        })
        mm.asset_map = {'USD': 'USD', 'EUR': 'EUR'}

        output_path = tmp_path / "prices.csv"
        mm.to_disk(str(output_path))

        assert output_path.exists()
        asset_map_path = output_path.parent / "prices_asset_map.csv"
        assert asset_map_path.exists()

    def test_load_pregenerated_data(self, tmp_path):
        """Test loading pre-generated price data."""
        mm = market_manager.MarketManager(ref_currency='GBP')
        
        # Create test data files
        prices_df = pd.DataFrame({
            'AssetMapped': ['USD', 'EUR'],
            'Date': ['2023-01-01', '2023-01-02'],
            'Price': [1.2, 1.1],
        })
        prices_df.to_csv(tmp_path / "prices.csv", index=False)

        asset_map_df = pd.DataFrame({
            'AssetMap': ['USD', 'EUR'],
        }, index=['USD', 'EUR'])
        asset_map_df.to_csv(tmp_path / "prices_asset_map.csv")

        mm.load_pregenerated_data(str(tmp_path / "prices.csv"))

        assert mm.prices is not None
        assert mm.asset_map is not None
        assert mm.asset_map['USD'] == 'USD'


class TestTransformationManager:
    """Tests for TransformationManager class."""

    @staticmethod
    def _create_sample_transactions(num_rows=5):
        """Helper to create sample transactions DataFrame."""
        return pd.DataFrame({
            'Account': ['Account1', 'Account1', 'Account2', 'Account1', 'Account2'],
            'Asset': ['GBP', 'GBP', 'USD', 'EUR', 'USD'],
            'Date': pd.date_range('2023-01-01', periods=num_rows),
            'Amount': [100, 50, 200, 75, 150],
            'Quantity': [100, 50, 200, 75, 150],
            'MemoMapped': ['memo1', 'memo2', 'memo3', 'memo4', 'memo5'],
            'Type': ['type1', 'type1', 'type2', 'type3', 'type2'],
            'SubType': ['subtype1', 'subtype1', 'subtype2', 'subtype3', 'subtype2'],
            'AssetPriceInRefCurrency': [1.0, 1.0, 1.25, 1.1, 1.25],
            'AssetPriceChangeInRefCurrency': [0.0, 0.0, 0.05, 0.02, 0.05],
            'FullMasterType': ['Income', 'Income', 'Expense', 'Expense', 'Expense'],
            'FullType': ['Salary', 'Salary', 'Gas', 'Electric', 'Gas'],
            'FullSubType': ['Base', 'Bonus', 'Petrol', 'Monthly', 'Petrol'],
            'Subcategory': ['test', 'test', 'test', 'test', 'test'],
            'Memo': ['memo1', 'memo2', 'memo3', 'memo4', 'memo5'],
            'Currency': ['GBP', 'GBP', 'USD', 'EUR', 'USD'],
            'MemoSimple': ['simple1', 'simple2', 'simple3', 'simple4', 'simple5'],
            'AccountType': ['type_a', 'type_a', 'type_b', 'type_a', 'type_b'],
            'FacingAccount': ['', '', '', '', ''],
            'SourceFile': ['file1', 'file1', 'file2', 'file1', 'file2'],
        })

    def test_init(self, config):
        """Test TransformationManager initialization."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        # Create minimal transactions data
        dm.transactions = self._create_sample_transactions()
        
        # Mock the asset map and prices
        mm.asset_map = {'GBP': 'GBP', 'USD': 'USD', 'EUR': 'EUR'}
        mm.prices = pd.DataFrame({
            'AssetMapped': ['GBP', 'USD', 'EUR'],
            'Date': pd.date_range('2023-01-01', periods=3),
            'AssetPriceInRefCurrency': [1.0, 1.25, 1.1],
            'AssetPriceChangeInRefCurrency': [0.0, 0.05, 0.02],
        }).set_index(['AssetMapped', 'Date'])
        
        tm = transformation_manager.TransformationManager(dm, mm)
        
        assert tm.data_manager is dm
        assert tm.market_manager is mm
        assert tm._df_grouped_transactions is not None

    def test_group_transaction(self, config):
        """Test transaction grouping functionality."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        dm.transactions = self._create_sample_transactions()
        mm.asset_map = {'GBP': 'GBP', 'USD': 'USD', 'EUR': 'EUR'}
        mm.prices = pd.DataFrame({
            'AssetMapped': ['GBP', 'USD', 'EUR'],
            'Date': pd.date_range('2023-01-01', periods=3),
            'AssetPriceInRefCurrency': [1.0, 1.25, 1.1],
            'AssetPriceChangeInRefCurrency': [0.0, 0.05, 0.02],
        }).set_index(['AssetMapped', 'Date'])
        
        tm = transformation_manager.TransformationManager(dm, mm)
        
        assert tm._df_grouped_transactions is not None
        assert isinstance(tm._df_grouped_transactions, pd.DataFrame)
        assert len(tm._df_grouped_transactions) > 0
        assert 'AssetMapped' in tm._df_grouped_transactions.index.names

    def test_get_values_by_asset_no_filters(self, config):
        """Test get_values_by_asset without filters."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        dm.transactions = self._create_sample_transactions()
        mm.asset_map = {'GBP': 'GBP', 'USD': 'USD', 'EUR': 'EUR'}
        mm.prices = pd.DataFrame({
            'AssetMapped': ['GBP', 'USD', 'EUR'],
            'Date': pd.date_range('2023-01-01', periods=3),
            'AssetPriceInRefCurrency': [1.0, 1.25, 1.1],
            'AssetPriceChangeInRefCurrency': [0.0, 0.05, 0.02],
        }).set_index(['AssetMapped', 'Date'])
        
        tm = transformation_manager.TransformationManager(dm, mm)
        values = tm.get_values_by_asset()
        
        assert isinstance(values, pd.DataFrame)
        assert 'Value' in values.columns
        assert 'CapitalGain' in values.columns

    def test_get_values_by_asset_with_date_range(self, config):
        """Test get_values_by_asset with date range filter."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        dm.transactions = self._create_sample_transactions()
        mm.asset_map = {'GBP': 'GBP', 'USD': 'USD', 'EUR': 'EUR'}
        mm.prices = pd.DataFrame({
            'AssetMapped': ['GBP', 'USD', 'EUR'],
            'Date': pd.date_range('2023-01-01', periods=3),
            'AssetPriceInRefCurrency': [1.0, 1.25, 1.1],
            'AssetPriceChangeInRefCurrency': [0.0, 0.05, 0.02],
        }).set_index(['AssetMapped', 'Date'])
        
        tm = transformation_manager.TransformationManager(dm, mm)
        date_range = [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-03')]
        values = tm.get_values_by_asset(date_range=date_range)
        
        assert isinstance(values, pd.DataFrame)
        assert len(values) <= len(tm.get_values_by_asset())

    def test_get_values_by_asset_with_account_filter(self, config):
        """Test get_values_by_asset with account filter."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        dm.transactions = self._create_sample_transactions()
        mm.asset_map = {'GBP': 'GBP', 'USD': 'USD', 'EUR': 'EUR'}
        mm.prices = pd.DataFrame({
            'AssetMapped': ['GBP', 'USD', 'EUR'],
            'Date': pd.date_range('2023-01-01', periods=3),
            'AssetPriceInRefCurrency': [1.0, 1.25, 1.1],
            'AssetPriceChangeInRefCurrency': [0.0, 0.05, 0.02],
        }).set_index(['AssetMapped', 'Date'])
        
        tm = transformation_manager.TransformationManager(dm, mm)
        values = tm.get_values_by_asset(account='Account1')
        
        assert isinstance(values, pd.DataFrame)

    def test_get_values_by_asset_invalid_date_range(self, config):
        """Test get_values_by_asset raises error with invalid date range."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        dm.transactions = self._create_sample_transactions()
        mm.asset_map = {'GBP': 'GBP', 'USD': 'USD', 'EUR': 'EUR'}
        mm.prices = pd.DataFrame({
            'AssetMapped': ['GBP', 'USD', 'EUR'],
            'Date': pd.date_range('2023-01-01', periods=3),
            'AssetPriceInRefCurrency': [1.0, 1.25, 1.1],
            'AssetPriceChangeInRefCurrency': [0.0, 0.05, 0.02],
        }).set_index(['AssetMapped', 'Date'])
        
        tm = transformation_manager.TransformationManager(dm, mm)
        with pytest.raises(ValueError):
            tm.get_values_by_asset(date_range=[pd.Timestamp('2023-01-01')])

    def test_consolidate_transactions(self):
        """Test consolidate_transactions static method."""
        transactions = [
            ('memo1', 100, 'type1', 'subtype1'),
            ('memo2', 50, 'type2', 'subtype2'),
            ('memo1', 30, 'type1', 'subtype1'),
        ]
        result = transformation_manager.TransformationManager.consolidate_transactions(transactions)
        
        assert len(result) == 2
        # 'memo1' should be aggregated: 100 + 30 = 130
        assert result[0][0] == 'memo1'
        assert result[0][1] == 130

    def test_aggregate_transactions(self):
        """Test aggregate_transactions static method."""
        transactions_list = [
            {'memo1': (100, 'type1', 'subtype1'), 'memo2': (50, 'type2', 'subtype2')},
            {'memo1': (30, 'type1', 'subtype1'), 'memo3': (20, 'type3', 'subtype3')},
        ]
        result = transformation_manager.TransformationManager.aggregate_transactions(transactions_list)
        
        assert isinstance(result, list)
        assert len(result) > 0

    def test_to_agg_dict(self):
        """Test to_agg_dict static method."""
        m_list = ['memo1', 'memo2', 'memo1']
        q_list = [100, 50, 30]
        p = 1.25
        t_list = ['type1', 'type2', 'type1']
        s_list = ['subtype1', 'subtype2', 'subtype1']
        
        result = transformation_manager.TransformationManager.to_agg_dict(m_list, q_list, p, t_list, s_list)
        
        assert isinstance(result, dict)
        assert 'memo1' in result
        assert result['memo1'][0] == (100 + 30) * p

    def test_get_values_timeseries(self, config):
        """Test get_values_timeseries functionality."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        dm.transactions = self._create_sample_transactions()
        mm.asset_map = {'GBP': 'GBP', 'USD': 'USD', 'EUR': 'EUR'}
        mm.prices = pd.DataFrame({
            'AssetMapped': ['GBP', 'USD', 'EUR'],
            'Date': pd.date_range('2023-01-01', periods=3),
            'AssetPriceInRefCurrency': [1.0, 1.25, 1.1],
            'AssetPriceChangeInRefCurrency': [0.0, 0.05, 0.02],
        }).set_index(['AssetMapped', 'Date'])
        
        tm = transformation_manager.TransformationManager(dm, mm)
        timeseries = tm.get_values_timeseries()
        
        assert isinstance(timeseries, pd.DataFrame)
        assert 'Value' in timeseries.columns
        assert 'TransactionValue_list' in timeseries.columns

    def test_get_flow_values_both(self, config):
        """Test get_flow_values with 'both' parameter."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        dm.transactions = self._create_sample_transactions()
        mm.asset_map = {'GBP': 'GBP', 'USD': 'USD', 'EUR': 'EUR'}
        mm.prices = pd.DataFrame({
            'AssetMapped': ['GBP', 'USD', 'EUR'],
            'Date': pd.date_range('2023-01-01', periods=3),
            'AssetPriceInRefCurrency': [1.0, 1.25, 1.1],
            'AssetPriceChangeInRefCurrency': [0.0, 0.05, 0.02],
        }).set_index(['AssetMapped', 'Date'])
        
        tm = transformation_manager.TransformationManager(dm, mm)
        flow_values = tm.get_flow_values()
        
        assert isinstance(flow_values, pd.DataFrame)

    def test_get_flow_values_with_dates(self, config):
        """Test get_flow_values with date range."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        dm.transactions = self._create_sample_transactions()
        mm.asset_map = {'GBP': 'GBP', 'USD': 'USD', 'EUR': 'EUR'}
        mm.prices = pd.DataFrame({
            'AssetMapped': ['GBP', 'USD', 'EUR'],
            'Date': pd.date_range('2023-01-01', periods=3),
            'AssetPriceInRefCurrency': [1.0, 1.25, 1.1],
            'AssetPriceChangeInRefCurrency': [0.0, 0.05, 0.02],
        }).set_index(['AssetMapped', 'Date'])
        
        tm = transformation_manager.TransformationManager(dm, mm)
        date_start = pd.Timestamp('2023-01-01')
        date_end = pd.Timestamp('2023-01-03')
        flow_values = tm.get_flow_values(date_start, date_end)
        
        assert isinstance(flow_values, pd.DataFrame)

    def test_get_flow_values_invalid_dates(self, config):
        """Test get_flow_values raises error with mismatched date types."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        dm.transactions = self._create_sample_transactions()
        mm.asset_map = {'GBP': 'GBP', 'USD': 'USD', 'EUR': 'EUR'}
        mm.prices = pd.DataFrame({
            'AssetMapped': ['GBP', 'USD', 'EUR'],
            'Date': pd.date_range('2023-01-01', periods=3),
            'AssetPriceInRefCurrency': [1.0, 1.25, 1.1],
            'AssetPriceChangeInRefCurrency': [0.0, 0.05, 0.02],
        }).set_index(['AssetMapped', 'Date'])
        
        tm = transformation_manager.TransformationManager(dm, mm)
        with pytest.raises(ValueError):
            tm.get_flow_values(date_start=pd.Timestamp('2023-01-01'))

    def test_get_flow_values_invalid_how(self, config):
        """Test get_flow_values raises error with invalid 'how' parameter."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        dm.transactions = self._create_sample_transactions()
        mm.asset_map = {'GBP': 'GBP', 'USD': 'USD', 'EUR': 'EUR'}
        mm.prices = pd.DataFrame({
            'AssetMapped': ['GBP', 'USD', 'EUR'],
            'Date': pd.date_range('2023-01-01', periods=3),
            'AssetPriceInRefCurrency': [1.0, 1.25, 1.1],
            'AssetPriceChangeInRefCurrency': [0.0, 0.05, 0.02],
        }).set_index(['AssetMapped', 'Date'])
        
        tm = transformation_manager.TransformationManager(dm, mm)
        with pytest.raises(ValueError):
            tm.get_flow_values(date_start=pd.Timestamp('2023-01-01'), 
                              date_end=pd.Timestamp('2023-01-03'),
                              how='invalid')

    def test_get_flow_values_in_filter(self, config):
        """Test get_flow_values with 'in' filter."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        dm.transactions = self._create_sample_transactions()
        mm.asset_map = {'GBP': 'GBP', 'USD': 'USD', 'EUR': 'EUR'}
        mm.prices = pd.DataFrame({
            'AssetMapped': ['GBP', 'USD', 'EUR'],
            'Date': pd.date_range('2023-01-01', periods=3),
            'AssetPriceInRefCurrency': [1.0, 1.25, 1.1],
            'AssetPriceChangeInRefCurrency': [0.0, 0.05, 0.02],
        }).set_index(['AssetMapped', 'Date'])
        
        tm = transformation_manager.TransformationManager(dm, mm)
        flow_values = tm.get_flow_values(how='in')
        
        assert isinstance(flow_values, pd.DataFrame)
        if len(flow_values) > 0:
            assert all(flow_values['Value'] > 0)

    def test_get_flow_values_out_filter(self, config):
        """Test get_flow_values with 'out' filter."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        dm.transactions = self._create_sample_transactions()
        mm.asset_map = {'GBP': 'GBP', 'USD': 'USD', 'EUR': 'EUR'}
        mm.prices = pd.DataFrame({
            'AssetMapped': ['GBP', 'USD', 'EUR'],
            'Date': pd.date_range('2023-01-01', periods=3),
            'AssetPriceInRefCurrency': [1.0, 1.25, 1.1],
            'AssetPriceChangeInRefCurrency': [0.0, 0.05, 0.02],
        }).set_index(['AssetMapped', 'Date'])
        
        tm = transformation_manager.TransformationManager(dm, mm)
        flow_values = tm.get_flow_values(how='out')
        
        assert isinstance(flow_values, pd.DataFrame)


class TestTransformationManagerCache:
    """Tests for TransformationManagerCache class."""

    @staticmethod
    def _create_sample_transactions(num_rows=5):
        """Helper to create sample transactions DataFrame."""
        return pd.DataFrame({
            'Account': ['Account1', 'Account1', 'Account2', 'Account1', 'Account2'],
            'Asset': ['GBP', 'GBP', 'USD', 'EUR', 'USD'],
            'Date': pd.date_range('2023-01-01', periods=num_rows),
            'Amount': [100, 50, 200, 75, 150],
            'Quantity': [100, 50, 200, 75, 150],
            'MemoMapped': ['memo1', 'memo2', 'memo3', 'memo4', 'memo5'],
            'Type': ['type1', 'type1', 'type2', 'type3', 'type2'],
            'SubType': ['subtype1', 'subtype1', 'subtype2', 'subtype3', 'subtype2'],
            'AssetPriceInRefCurrency': [1.0, 1.0, 1.25, 1.1, 1.25],
            'AssetPriceChangeInRefCurrency': [0.0, 0.0, 0.05, 0.02, 0.05],
            'FullMasterType': ['Income', 'Income', 'Expense', 'Expense', 'Expense'],
            'FullType': ['Salary', 'Salary', 'Gas', 'Electric', 'Gas'],
            'FullSubType': ['Base', 'Bonus', 'Petrol', 'Monthly', 'Petrol'],
            'Subcategory': ['test', 'test', 'test', 'test', 'test'],
            'Memo': ['memo1', 'memo2', 'memo3', 'memo4', 'memo5'],
            'Currency': ['GBP', 'GBP', 'USD', 'EUR', 'USD'],
            'MemoSimple': ['simple1', 'simple2', 'simple3', 'simple4', 'simple5'],
            'AccountType': ['type_a', 'type_a', 'type_b', 'type_a', 'type_b'],
            'FacingAccount': ['', '', '', '', ''],
            'SourceFile': ['file1', 'file1', 'file2', 'file1', 'file2'],
        })

    def test_init_with_defaults(self, config):
        """Test TransformationManagerCache initialization with defaults."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        dm.transactions = self._create_sample_transactions()
        mm.asset_map = {'GBP': 'GBP', 'USD': 'USD', 'EUR': 'EUR'}
        mm.prices = pd.DataFrame({
            'AssetMapped': ['GBP', 'USD', 'EUR'],
            'Date': pd.date_range('2023-01-01', periods=3),
            'AssetPriceInRefCurrency': [1.0, 1.25, 1.1],
            'AssetPriceChangeInRefCurrency': [0.0, 0.05, 0.02],
        }).set_index(['AssetMapped', 'Date'])
        
        tmc = TransformationManagerCache(dm, mm, year=2023)
        
        assert tmc._account is None
        assert tmc._year == 2023
        assert tmc._hows == ['both']
        assert tmc._include_iat == False
        assert tmc._include_full_types == True

    def test_init_with_custom_params(self, config):
        """Test TransformationManagerCache initialization with custom parameters."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        dm.transactions = self._create_sample_transactions()
        mm.asset_map = {'GBP': 'GBP', 'USD': 'USD', 'EUR': 'EUR'}
        mm.prices = pd.DataFrame({
            'AssetMapped': ['GBP', 'USD', 'EUR'],
            'Date': pd.date_range('2023-01-01', periods=3),
            'AssetPriceInRefCurrency': [1.0, 1.25, 1.1],
            'AssetPriceChangeInRefCurrency': [0.0, 0.05, 0.02],
        }).set_index(['AssetMapped', 'Date'])
        
        tmc = TransformationManagerCache(dm, mm, year=2023, account='Account1', hows=['in', 'out'])
        
        assert tmc._account == 'Account1'
        assert tmc._hows == ['in', 'out']

    def test_get_flow_values_from_cache(self, config):
        """Test get_flow_values retrieves from cache."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        dm.transactions = self._create_sample_transactions()
        mm.asset_map = {'GBP': 'GBP', 'USD': 'USD', 'EUR': 'EUR'}
        mm.prices = pd.DataFrame({
            'AssetMapped': ['GBP', 'USD', 'EUR'],
            'Date': pd.date_range('2023-01-01', periods=3),
            'AssetPriceInRefCurrency': [1.0, 1.25, 1.1],
            'AssetPriceChangeInRefCurrency': [0.0, 0.05, 0.02],
        }).set_index(['AssetMapped', 'Date'])
        
        tmc = TransformationManagerCache(dm, mm, year=2023)
        
        # Should return copy from cache
        flow_values = tmc.get_flow_values()
        assert isinstance(flow_values, pd.DataFrame)

    def test_get_flow_values_invalid_account(self, config):
        """Test get_flow_values raises error with invalid account."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        dm.transactions = self._create_sample_transactions()
        mm.asset_map = {'GBP': 'GBP', 'USD': 'USD', 'EUR': 'EUR'}
        mm.prices = pd.DataFrame({
            'AssetMapped': ['GBP', 'USD', 'EUR'],
            'Date': pd.date_range('2023-01-01', periods=3),
            'AssetPriceInRefCurrency': [1.0, 1.25, 1.1],
            'AssetPriceChangeInRefCurrency': [0.0, 0.05, 0.02],
        }).set_index(['AssetMapped', 'Date'])
        
        tmc = TransformationManagerCache(dm, mm, year=2023, account='Account1')
        
        with pytest.raises(ValueError):
            tmc.get_flow_values(date_start=pd.Timestamp('2023-01-01'), date_end=pd.Timestamp('2023-01-03'), account='Account2')

    def test_get_flow_values_invalid_how(self, config):
        """Test get_flow_values raises error with non-cached 'how' parameter."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        dm.transactions = self._create_sample_transactions()
        mm.asset_map = {'GBP': 'GBP', 'USD': 'USD', 'EUR': 'EUR'}
        mm.prices = pd.DataFrame({
            'AssetMapped': ['GBP', 'USD', 'EUR'],
            'Date': pd.date_range('2023-01-01', periods=3),
            'AssetPriceInRefCurrency': [1.0, 1.25, 1.1],
            'AssetPriceChangeInRefCurrency': [0.0, 0.05, 0.02],
        }).set_index(['AssetMapped', 'Date'])
        
        tmc = TransformationManagerCache(dm, mm, year=2023, hows=['both'])
        
        with pytest.raises(ValueError):
            tmc.get_flow_values(date_start=pd.Timestamp('2023-01-01'), date_end=pd.Timestamp('2023-01-03'), how='in')

    def test_get_flow_values_invalid_include_iat(self, config):
        """Test get_flow_values raises error with mismatched include_iat."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        dm.transactions = self._create_sample_transactions()
        mm.asset_map = {'GBP': 'GBP', 'USD': 'USD', 'EUR': 'EUR'}
        mm.prices = pd.DataFrame({
            'AssetMapped': ['GBP', 'USD', 'EUR'],
            'Date': pd.date_range('2023-01-01', periods=3),
            'AssetPriceInRefCurrency': [1.0, 1.25, 1.1],
            'AssetPriceChangeInRefCurrency': [0.0, 0.05, 0.02],
        }).set_index(['AssetMapped', 'Date'])
        
        tmc = TransformationManagerCache(dm, mm, year=2023, include_iat=False)
        
        with pytest.raises(ValueError):
            tmc.get_flow_values(date_start=pd.Timestamp('2023-01-01'), date_end=pd.Timestamp('2023-01-03'), include_iat=True)
