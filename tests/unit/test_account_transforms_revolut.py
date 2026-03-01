"""Tests for Revolut account transform (simplified to avoid CSV format complexities)."""
import os
import pytest
import pandas as pd
import configparser
from bkanalysis.transforms.account_transforms import revolut_transform


@pytest.fixture
def revolut_config():
    """Provide Revolut transform configuration."""
    config = configparser.ConfigParser()
    config['revolut'] = {
        'expected_columns': "['Completed Date', 'Description', 'Paid Out (CCY)', 'Paid In (CCY)', 'Exchange Out', 'Exchange In', 'Balance (CCY)', 'Category', 'Notes']",
        'possible_currencies': "['GBP', 'EUR', 'USD']",
        'account_name': 'Revolut',
        'account_type': 'current',
    }
    return config['revolut']


@pytest.fixture
def revolut_gbp_csv_file(tmp_path):
    """Create a valid Revolut GBP CSV file with proper spacing."""
    # Revolut uses space " " as separator in the actual files
    csv_content = """Completed Date , Description , Paid Out (GBP) , Paid In (GBP) , Exchange Out , Exchange In , Balance (GBP) , Category , Notes
1 Jan 2024 , Transfer In ,  , 2500.00 ,  ,  , 5000.00 , Transfer , Payment received
2 Jan 2024 , Vendor Payment , 10.50 ,  ,  ,  , 4500.00 , Purchase , Merchant
3 Jan 2024 , Utility Payment , 150.00 ,  ,  ,  , 4510.50 , Bill , Service
4 Jan 2024 , Cash Withdrawal , 200.00 ,  ,  ,  , 4660.50 , Cash , Withdrawal
"""
    csv_path = os.path.join(tmp_path, "revolut_statement.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


class TestRevolutTransform:
    """Tests for Revolut account transform."""

    def test_can_handle_valid_revolut_gbp_csv(self, revolut_gbp_csv_file, revolut_config):
        """Test that can_handle recognizes valid Revolut GBP CSV with proper separator."""
        # Revolut uses " , " (space-comma-space) as separator
        assert revolut_transform.can_handle(revolut_gbp_csv_file, revolut_config, sep=' , ') is True

    def test_can_handle_invalid_extension(self, tmp_path, revolut_config):
        """Test that can_handle rejects non-CSV files."""
        txt_file = os.path.join(tmp_path, "statement.txt")
        with open(txt_file, 'w') as f:
            f.write("Completed Date , Description")
        assert revolut_transform.can_handle(txt_file, revolut_config, sep=' , ') is False

    def test_load_valid_revolut_gbp_csv(self, revolut_gbp_csv_file, revolut_config):
        """Test loading a valid Revolut GBP CSV."""
        result = revolut_transform.load(revolut_gbp_csv_file, revolut_config, sep=' , ')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert all(col in result.columns for col in ['Date', 'Account', 'Amount', 'Subcategory', 'Memo', 'Currency', 'AccountType'])

    def test_load_date_parsing(self, revolut_gbp_csv_file, revolut_config):
        """Test that dates are correctly parsed."""
        result = revolut_transform.load(revolut_gbp_csv_file, revolut_config, sep=' , ')
        assert pd.api.types.is_datetime64_any_dtype(result.Date)

    def test_load_currency_detection_gbp(self, revolut_gbp_csv_file, revolut_config):
        """Test that currency is detected from column names."""
        result = revolut_transform.load(revolut_gbp_csv_file, revolut_config, sep=' , ')
        assert (result.Currency == 'GBP').all()

    def test_load_account_name(self, revolut_gbp_csv_file, revolut_config):
        """Test that Account is set correctly."""
        result = revolut_transform.load(revolut_gbp_csv_file, revolut_config, sep=' , ')
        # Account should be set from config or similar
        assert 'Revolut' in str(result.Account.iloc[0])

    def test_load_account_type(self, revolut_gbp_csv_file, revolut_config):
        """Test that AccountType is set correctly."""
        result = revolut_transform.load(revolut_gbp_csv_file, revolut_config, sep=' , ')
        assert (result.AccountType == 'current').all()

    def test_load_memo_from_description(self, revolut_gbp_csv_file, revolut_config):
        """Test that Memo contains description."""
        result = revolut_transform.load(revolut_gbp_csv_file, revolut_config, sep=' , ')
        assert 'Transfer' in result.Memo.iloc[0]
        assert 'Vendor' in result.Memo.iloc[1]

    def test_load_has_all_target_columns(self, revolut_gbp_csv_file, revolut_config):
        """Test that all target columns are present."""
        result = revolut_transform.load(revolut_gbp_csv_file, revolut_config, sep=' , ')
        assert all(col in result.columns for col in ['Date', 'Account', 'Amount', 'Subcategory', 'Memo', 'Currency', 'AccountType'])

    def test_load_preserves_all_transactions(self, revolut_gbp_csv_file, revolut_config):
        """Test that all transactions are preserved."""
        result = revolut_transform.load(revolut_gbp_csv_file, revolut_config, sep=' , ')
        assert len(result) == 4
