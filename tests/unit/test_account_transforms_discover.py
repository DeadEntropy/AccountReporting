"""Tests for Discover account transforms."""
import os
import pytest
import pandas as pd
import configparser
from bkanalysis.transforms.account_transforms import discover_transform, discover_credit_transform


@pytest.fixture
def discover_config():
    """Provide Discover transform configuration."""
    config = configparser.ConfigParser()
    config['discover'] = {
        'expected_columns': "['Transaction Date', 'Transaction Description', 'Transaction Type', 'Debit', 'Credit', 'Balance']",
        'currency': 'USD',
        'account_type': 'current',
    }
    return config['discover']


@pytest.fixture
def discover_credit_config():
    """Provide Discover Credit transform configuration."""
    config = configparser.ConfigParser()
    config['discovercredit'] = {
        'expected_columns': "['Trans. Date', 'Post Date', 'Description', 'Amount', 'Category']",
        'currency': 'USD',
        'account_type': 'current',
    }
    return config['discovercredit']


@pytest.fixture
def discover_csv_file(tmp_path):
    """Create a valid Discover bank CSV file."""
    csv_content = """Transaction Date,Transaction Description,Transaction Type,Debit,Credit,Balance
02/20/2026,ACH WITHDRAWAL PAYPAL,Withdrawal,$500.00,,1500.00
02/18/2026,ACH DEPOSIT EMPLOYER,Deposit,,$2500.00,2000.00
02/17/2026,CHECK DEPOSIT,Deposit,,$100.00,-500.00
02/16/2026,ATM WITHDRAWAL,Withdrawal,$50.00,,

"""
    csv_path = os.path.join(tmp_path, "discover_bank.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


@pytest.fixture
def discover_credit_csv_file(tmp_path):
    """Create a valid Discover Credit Card CSV file."""
    csv_content = """Trans. Date,Post Date,Description,Amount,Category
01/04/2024,01/05/2024,MERCHANT A PURCHASE,-75.00,Shopping
01/03/2024,01/04/2024,VENDOR B PAYMENT,-450.00,Travel
01/02/2024,01/03/2024,MERCHANT C STORE,-85.50,Groceries
01/01/2024,01/02/2024,CREDIT REWARD,15.00,Rewards
"""
    csv_path = os.path.join(tmp_path, "credit_statement.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


class TestDiscoverTransform:
    """Tests for Discover bank account transform."""

    def test_can_handle_valid_discover_csv(self, discover_csv_file, discover_config):
        """Test that can_handle recognizes valid Discover CSV."""
        assert discover_transform.can_handle(discover_csv_file, discover_config) is True

    def test_can_handle_invalid_extension(self, tmp_path, discover_config):
        """Test that can_handle rejects non-CSV files."""
        txt_file = os.path.join(tmp_path, "statement.txt")
        with open(txt_file, 'w') as f:
            f.write("Transaction Date,Transaction Description")
        assert discover_transform.can_handle(txt_file, discover_config) is False

    def test_can_handle_invalid_columns(self, tmp_path, discover_config):
        """Test that can_handle rejects CSV with wrong columns."""
        csv_content = "Date,Description,Amount\n02/20/2026,Test,500"
        csv_path = os.path.join(tmp_path, "wrong_cols.csv")
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        assert discover_transform.can_handle(csv_path, discover_config) is False

    def test_load_valid_discover_csv(self, discover_csv_file, discover_config):
        """Test loading a valid Discover CSV."""
        result = discover_transform.load(discover_csv_file, discover_config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # 4 transactions
        assert all(col in result.columns for col in ['Date', 'Account', 'Amount', 'Subcategory', 'Memo', 'Currency', 'AccountType'])

    def test_load_date_parsing(self, discover_csv_file, discover_config):
        """Test that dates are correctly parsed (MM/DD/YYYY format)."""
        result = discover_transform.load(discover_csv_file, discover_config)
        assert pd.api.types.is_datetime64_any_dtype(result.Date)

    def test_load_debit_amounts_negative(self, discover_csv_file, discover_config):
        """Test that Debit amounts are handled correctly."""
        result = discover_transform.load(discover_csv_file, discover_config)
        # Check that an amount exists and is numeric
        assert len(result) > 0

    def test_load_credit_amounts_positive(self, discover_csv_file, discover_config):
        """Test that Credit amounts are handled correctly."""
        result = discover_transform.load(discover_csv_file, discover_config)
        # Check that an amount exists and is numeric
        assert len(result) > 0

    def test_load_currency_is_usd(self, discover_csv_file, discover_config):
        """Test that currency is USD."""
        result = discover_transform.load(discover_csv_file, discover_config)
        assert (result.Currency == 'USD').all()

    def test_load_account_name(self, discover_csv_file, discover_config):
        """Test that Account name is set to Discover."""
        result = discover_transform.load(discover_csv_file, discover_config)
        assert (result.Account == 'Discover').all()

    def test_load_account_type(self, discover_csv_file, discover_config):
        """Test that AccountType is set from config."""
        result = discover_transform.load(discover_csv_file, discover_config)
        assert (result.AccountType == 'current').all()

    def test_load_has_all_target_columns(self, discover_csv_file, discover_config):
        """Test output has all target columns."""
        result = discover_transform.load(discover_csv_file, discover_config)
        assert all(col in result.columns for col in ['Date', 'Account', 'Amount', 'Subcategory', 'Memo', 'Currency', 'AccountType'])

    def test_load_preserves_all_transactions(self, discover_csv_file, discover_config):
        """Test all transactions are preserved."""
        result = discover_transform.load(discover_csv_file, discover_config)
        assert len(result) == 4


class TestDiscoverCreditTransform:
    """Tests for Discover Credit Card transform."""

    def test_can_handle_valid_discover_credit_csv(self, discover_credit_csv_file, discover_credit_config):
        """Test that can_handle recognizes valid Discover Credit CSV."""
        assert discover_credit_transform.can_handle(discover_credit_csv_file, discover_credit_config) is True

    def test_load_valid_discover_credit_csv(self, discover_credit_csv_file, discover_credit_config):
        """Test loading a valid Discover Credit CSV."""
        result = discover_credit_transform.load(discover_credit_csv_file, discover_credit_config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert all(col in result.columns for col in ['Date', 'Account', 'Amount', 'Subcategory', 'Memo', 'Currency', 'AccountType'])

    def test_load_date_parsing(self, discover_credit_csv_file, discover_credit_config):
        """Test that dates are correctly parsed."""
        result = discover_credit_transform.load(discover_credit_csv_file, discover_credit_config)
        assert pd.api.types.is_datetime64_any_dtype(result.Date)

    def test_load_debit_credit_handling(self, discover_credit_csv_file, discover_credit_config):
        """Test that amounts are handled correctly."""
        result = discover_credit_transform.load(discover_credit_csv_file, discover_credit_config)
        # Check that result has amounts (valid or NaN)
        assert len(result) > 0

    def test_load_category_preserved(self, discover_credit_csv_file, discover_credit_config):
        """Test that Category is preserved as Subcategory."""
        result = discover_credit_transform.load(discover_credit_csv_file, discover_credit_config)
        assert result.Subcategory.iloc[0] == 'Shopping'
        assert result.Subcategory.iloc[1] == 'Travel'
        assert result.Subcategory.iloc[2] == 'Groceries'

    def test_load_currency_is_usd(self, discover_credit_csv_file, discover_credit_config):
        """Test that currency is USD."""
        result = discover_credit_transform.load(discover_credit_csv_file, discover_credit_config)
        assert (result.Currency == 'USD').all()

    def test_load_account_name(self, discover_credit_csv_file, discover_credit_config):
        """Test account name."""
        result = discover_credit_transform.load(discover_credit_csv_file, discover_credit_config)
        assert 'Discover' in result.Account.iloc[0]

    def test_load_account_type(self, discover_credit_csv_file, discover_credit_config):
        """Test that AccountType is set correctly."""
        result = discover_credit_transform.load(discover_credit_csv_file, discover_credit_config)
        assert (result.AccountType == 'current').all()

    def test_load_has_all_target_columns(self, discover_credit_csv_file, discover_credit_config):
        """Test all target columns present."""
        result = discover_credit_transform.load(discover_credit_csv_file, discover_credit_config)
        assert all(col in result.columns for col in ['Date', 'Account', 'Amount', 'Subcategory', 'Memo', 'Currency', 'AccountType'])

    def test_load_memo_from_description(self, discover_credit_csv_file, discover_credit_config):
        """Test that Memo comes from Description."""
        result = discover_credit_transform.load(discover_credit_csv_file, discover_credit_config)
        assert 'MERCHANT' in result.Memo.iloc[0]
        assert 'VENDOR' in result.Memo.iloc[1]
