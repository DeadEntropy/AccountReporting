"""Tests for CoinbasePro account transform."""
import os
import pytest
import pandas as pd
import tempfile
import configparser

from bkanalysis.transforms.account_transforms import coinbase_pro_transform


@pytest.fixture
def coinbase_pro_config():
    """Provide CoinbasePro transform configuration."""
    config = configparser.ConfigParser()
    config['coinbasepro'] = {
        'expected_columns': "['portfolio', 'type', 'time', 'amount', 'balance', 'amount/balance unit', 'transfer id', 'trade id', 'order id']",
        'account_name': 'Coinbase Pro',
        'account_type': 'Crypto',
    }
    return config['coinbasepro']


@pytest.fixture
def coinbase_pro_csv_file(tmp_path):
    """Create a sample Coinbase Pro CSV file."""
    csv_content = """portfolio,type,time,amount,balance,amount/balance unit,transfer id,trade id,order id
default,deposit,2024-01-15T10:30:00Z,1.5,1.5,BTC,txn_12345,,
default,match,2024-01-16T14:20:00Z,-500,0.75,USD,,trade_67890,order_abc123
default,deposit,2024-01-17T08:15:00Z,0.5,1.25,BTC,txn_12346,,
"""
    csv_path = os.path.join(tmp_path, "cbpro_ledger.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


class TestCoinbaseProTransform:
    """Test CoinbasePro transform."""

    def test_can_handle_valid_coinbase_pro_csv(self, coinbase_pro_csv_file, coinbase_pro_config):
        """Test that can_handle recognizes valid CoinbasePro CSV."""
        assert coinbase_pro_transform.can_handle(coinbase_pro_csv_file, coinbase_pro_config)

    def test_can_handle_invalid_extension(self, tmp_path, coinbase_pro_config):
        """Test that can_handle rejects non-CSV files."""
        txt_file = tmp_path / "ledger.txt"
        txt_file.write_text("some text")
        assert not coinbase_pro_transform.can_handle(str(txt_file), coinbase_pro_config)

    def test_can_handle_invalid_columns(self, tmp_path, coinbase_pro_config):
        """Test that can_handle rejects files with wrong columns."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("id,amount,price\n1,1.5,25000")
        assert not coinbase_pro_transform.can_handle(str(bad_csv), coinbase_pro_config)

    def test_load_valid_coinbase_pro_csv(self, coinbase_pro_csv_file, coinbase_pro_config):
        """Test loading valid CoinbasePro CSV returns proper DataFrame."""
        result = coinbase_pro_transform.load(coinbase_pro_csv_file, coinbase_pro_config)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert all(col in result.columns for col in ['Date', 'Account', 'Currency', 'Amount', 'Memo'])

    def test_load_date_parsing(self, coinbase_pro_csv_file, coinbase_pro_config):
        """Test that ISO 8601 dates are properly parsed."""
        result = coinbase_pro_transform.load(coinbase_pro_csv_file, coinbase_pro_config)
        
        assert pd.api.types.is_datetime64_any_dtype(result['Date'])
        assert result.iloc[0]['Date'] == pd.Timestamp('2024-01-15')

    def test_load_account_name_set(self, coinbase_pro_csv_file, coinbase_pro_config):
        """Test that account name is set from config."""
        result = coinbase_pro_transform.load(coinbase_pro_csv_file, coinbase_pro_config)
        
        assert (result['Account'] == 'Coinbase Pro').all()

    def test_load_account_type_set(self, coinbase_pro_csv_file, coinbase_pro_config):
        """Test that account type is set from config."""
        result = coinbase_pro_transform.load(coinbase_pro_csv_file, coinbase_pro_config)
        
        assert (result['AccountType'] == 'Crypto').all()

    def test_load_currency_from_unit(self, coinbase_pro_csv_file, coinbase_pro_config):
        """Test that currency comes from amount/balance unit column."""
        result = coinbase_pro_transform.load(coinbase_pro_csv_file, coinbase_pro_config)
        
        assert result.iloc[0]['Currency'] == 'BTC'
        assert result.iloc[1]['Currency'] == 'USD'
        assert result.iloc[2]['Currency'] == 'BTC'

    def test_load_amount_preserved(self, coinbase_pro_csv_file, coinbase_pro_config):
        """Test that amounts are preserved."""
        result = coinbase_pro_transform.load(coinbase_pro_csv_file, coinbase_pro_config)
        
        assert result.iloc[0]['Amount'] == 1.5
        assert result.iloc[1]['Amount'] == -500
        assert result.iloc[2]['Amount'] == 0.5

    def test_load_memo_contains_type(self, coinbase_pro_csv_file, coinbase_pro_config):
        """Test that memo contains transaction type."""
        result = coinbase_pro_transform.load(coinbase_pro_csv_file, coinbase_pro_config)
        
        assert 'deposit' in result.iloc[0]['Memo'].lower()
        assert 'match' in result.iloc[1]['Memo'].lower()

    def test_load_has_all_target_columns(self, coinbase_pro_csv_file, coinbase_pro_config):
        """Test that output has all required target columns."""
        result = coinbase_pro_transform.load(coinbase_pro_csv_file, coinbase_pro_config)
        
        expected_cols = ['Date', 'Account', 'Currency', 'Amount', 'Memo', 'Subcategory', 'AccountType']
        assert all(col in result.columns for col in expected_cols)

    def test_load_preserves_all_transactions(self, coinbase_pro_csv_file, coinbase_pro_config):
        """Test that all transactions are preserved."""
        result = coinbase_pro_transform.load(coinbase_pro_csv_file, coinbase_pro_config)
        
        assert len(result) == 3
        assert result['Amount'].sum() == -498.0  # 1.5 - 500 + 0.5 = -498
