"""Tests for manager classes (data_manager, market_manager, transformation_manager)."""
import pytest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
from datetime import datetime

from bkanalysis.managers import data_manager, market_manager, transformation_manager


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

    def test_init(self, config):
        """Test TransformationManager initialization."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        # Create minimal transactions data to avoid full data loading
        dm.transactions = pd.DataFrame({
            'Account': ['Account1'],
            'Asset': ['GBP'],
            'Date': ['2023-01-01'],
            'Amount': [100],
            'Quantity': [100],
            'MemoMapped': ['test'],
            'Type': ['test'],
            'SubType': ['test'],
            'AssetPriceInRefCurrency': [1.0],
            'AssetPriceChangeInRefCurrency': [0.0],
        })
        
        # Mock the asset map
        mm.asset_map = {'GBP': 'GBP'}
        
        tm = transformation_manager.TransformationManager(dm, mm)
        
        assert tm.data_manager is dm
        assert tm.market_manager is mm
        assert tm._df_grouped_transactions is not None

    def test_group_transaction(self, config):
        """Test transaction grouping."""
        dm = data_manager.DataManager(config)
        mm = market_manager.MarketManager(ref_currency='GBP', config=config)
        
        # Create sample transactions
        dates = pd.date_range('2023-01-01', periods=3)
        dm.transactions = pd.DataFrame({
            'Account': ['Account1', 'Account1', 'Account2'],
            'Asset': ['GBP', 'GBP', 'USD'],
            'Date': dates,
            'Amount': [100, 50, 200],
            'Quantity': [100, 50, 200],
            'MemoMapped': ['test1', 'test2', 'test3'],
            'Type': ['type1', 'type1', 'type2'],
            'SubType': ['subtype1', 'subtype1', 'subtype2'],
            'AssetPriceInRefCurrency': [1.0, 1.0, 1.25],
            'AssetPriceChangeInRefCurrency': [0.0, 0.0, 0.05],
        })
        
        mm.asset_map = {'GBP': 'GBP', 'USD': 'USD'}
        
        tm = transformation_manager.TransformationManager(dm, mm)
        
        # Verify grouped transactions exist
        assert tm._df_grouped_transactions is not None
        assert len(tm._df_grouped_transactions) > 0
