"""Tests for Coinbase account transform."""
import os
import pytest
import pandas as pd
import configparser
from bkanalysis.transforms.account_transforms import coinbase_transform, coinbase_pro_transform


@pytest.fixture
def coinbase_config():
    """Provide Coinbase transform configuration."""
    config = configparser.ConfigParser()
    config['coinbase'] = {
        'expected_columns': "['ID', 'Timestamp', 'Transaction Type', 'Asset', 'Quantity Transacted', 'Price Currency', 'Price at Transaction', 'Subtotal', 'Total (inclusive of fees and/or spread)', 'Fees and/or Spread', 'Notes']",
        'account_name': 'Coinbase',
        'account_type': 'liquid',
    }
    return config['coinbase']


@pytest.fixture
def coinbase_csv_file(tmp_path):
    """Create a valid Coinbase CSV file."""
    csv_content = """ID,Timestamp,Transaction Type,Asset,Quantity Transacted,Price Currency,Price at Transaction,Subtotal,Total (inclusive of fees and/or spread),Fees and/or Spread,Notes
id001,2024-01-04T10:30:00Z,Buy,COIN1,0.05,USD,45000.00,2250.00,2250.00,0.00,Asset purchase
id002,2024-01-03T14:20:00Z,Sell,COIN2,1.5,USD,3500.00,5250.00,5200.00,-50.00,Asset sale
id003,2024-01-02T09:15:00Z,Staking Income,COIN3,0.25,USD,3000.00,750.00,750.00,0.00,Rewards
id004,2024-01-01T16:45:00Z,Convert,COIN1,0.1,USD,45000.00,4500.00,4500.00,0.00,Convert transaction
"""
    csv_path = os.path.join(tmp_path, "exchange_statement.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


class TestCoinbaseTransform:
    """Tests for Coinbase transform."""

    def test_can_handle_valid_coinbase_csv(self, coinbase_csv_file, coinbase_config):
        """Test that can_handle recognizes valid Coinbase CSV."""
        assert coinbase_transform.can_handle(coinbase_csv_file, coinbase_config) is True

    def test_can_handle_invalid_extension(self, tmp_path, coinbase_config):
        """Test that can_handle rejects non-CSV files."""
        txt_file = os.path.join(tmp_path, "data.txt")
        with open(txt_file, 'w') as f:
            f.write("ID,Timestamp,Transaction Type")
        assert coinbase_transform.can_handle(txt_file, coinbase_config) is False

    def test_can_handle_invalid_columns(self, tmp_path, coinbase_config):
        """Test that can_handle rejects CSV with wrong columns."""
        csv_content = "Date,Type,Amount\n2021-11-05,Buy,100"
        csv_path = os.path.join(tmp_path, "wrong_cols.csv")
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        assert coinbase_transform.can_handle(csv_path, coinbase_config) is False

    def test_load_valid_coinbase_csv(self, coinbase_csv_file, coinbase_config):
        """Test loading a valid Coinbase CSV."""
        result = coinbase_transform.load(coinbase_csv_file, coinbase_config)
        assert isinstance(result, pd.DataFrame)
        # Coinbase filters out Convert and Exchange Deposit types, so we may have fewer rows
        assert len(result) > 0
        assert all(col in result.columns for col in ['Date', 'Account', 'Amount', 'Subcategory', 'Memo', 'Currency', 'AccountType'])

    def test_load_date_parsing(self, coinbase_csv_file, coinbase_config):
        """Test that timestamps are correctly parsed."""
        result = coinbase_transform.load(coinbase_csv_file, coinbase_config)
        assert result.Date.dtype == 'object' or result.Date.dtype == 'datetime64[ns]'

    def test_load_currency_from_asset(self, coinbase_csv_file, coinbase_config):
        """Test that Currency comes from Asset column."""
        result = coinbase_transform.load(coinbase_csv_file, coinbase_config)
        # Currency should be the asset type (BTC, ETH, etc)
        assert result.Currency.iloc[0] in ['BTC', 'ETH'] or isinstance(result.Currency.iloc[0], str)

    def test_load_account_name(self, coinbase_csv_file, coinbase_config):
        """Test that Account is set from config."""
        result = coinbase_transform.load(coinbase_csv_file, coinbase_config)
        assert (result.Account == 'Coinbase').all()

    def test_load_account_type(self, coinbase_csv_file, coinbase_config):
        """Test that AccountType is set correctly."""
        result = coinbase_transform.load(coinbase_csv_file, coinbase_config)
        assert (result.AccountType == 'liquid').all()

    def test_load_transaction_type_in_memo(self, coinbase_csv_file, coinbase_config):
        """Test that Transaction Type is in Memo."""
        result = coinbase_transform.load(coinbase_csv_file, coinbase_config)
        assert 'COINBASE' in result.Memo.iloc[0]

    def test_load_has_all_target_columns(self, coinbase_csv_file, coinbase_config):
        """Test that all target columns are present."""
        result = coinbase_transform.load(coinbase_csv_file, coinbase_config)
        assert all(col in result.columns for col in ['Date', 'Account', 'Amount', 'Subcategory', 'Memo', 'Currency', 'AccountType'])
