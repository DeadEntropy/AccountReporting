"""Tests for Capital One Credit Card account transform."""
import os
import pytest
import pandas as pd
import configparser
from bkanalysis.transforms.account_transforms import capital_one_credit_transform, capital_one_transform


@pytest.fixture
def capital_one_credit_config():
    """Provide Capital One Credit transform configuration."""
    config = configparser.ConfigParser()
    config['capitalonecredit'] = {
        'expected_columns': "['Transaction Date', 'Posted Date', 'Card No.', 'Description', 'Category', 'Debit', 'Credit']",
        'currency': 'USD',
        'account_name': 'Discover Credit',
        'account_type': 'current',
    }
    return config['capitalonecredit']


@pytest.fixture
def capital_one_config():
    """Provide Capital One (360) transform configuration."""
    config = configparser.ConfigParser()
    config['capitalone'] = {
        'expected_columns': "['Account Number', 'Transaction Date', 'Transaction Amount', 'Transaction Type', 'Transaction Description', 'Balance']",
        'account_name': 'CapitalOne',
        'account_type': 'liquid',
    }
    return config['capitalone']


@pytest.fixture
def capital_one_credit_csv_file(tmp_path):
    """Create a valid Capital One Credit Card CSV file."""
    csv_content = """Transaction Date,Posted Date,Card No.,Description,Category,Debit,Credit
2026-10-28,2026-10-29,4807,GOOGLE *GOOGLE ONE,Internet,1.99,
2026-10-08,2026-10-09,4807,PAYPAL *DIMSUMSAM,Dining,80.70,
2026-10-01,2026-10-01,4807,DIRECTPAY FULL BALANCE,Payment/Credit,,615.17
2026-09-05,2026-09-06,0165,WL *STEAM PURCHASE,Internet,,26.65
"""
    csv_path = os.path.join(tmp_path, "CapitalOne_4807.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


class TestCapitalOneCreditTransform:
    """Tests for Capital One Credit Card transform."""

    def test_can_handle_valid_csv(self, capital_one_credit_csv_file, capital_one_credit_config):
        """Test that can_handle recognizes valid Capital One Credit CSV."""
        assert capital_one_credit_transform.can_handle(capital_one_credit_csv_file, capital_one_credit_config) is True

    def test_can_handle_invalid_extension(self, tmp_path, capital_one_credit_config):
        """Test that can_handle rejects non-CSV files."""
        txt_file = os.path.join(tmp_path, "statement.txt")
        with open(txt_file, 'w') as f:
            f.write("Transaction Date,Posted Date,Card No.,Description,Category,Debit,Credit")
        assert capital_one_credit_transform.can_handle(txt_file, capital_one_credit_config) is False

    def test_can_handle_invalid_columns(self, tmp_path, capital_one_credit_config):
        """Test that can_handle rejects CSV with wrong columns."""
        csv_path = os.path.join(tmp_path, "wrong_cols.csv")
        with open(csv_path, 'w') as f:
            f.write("Date,Description,Amount\n2026-10-28,Test,500")
        assert capital_one_credit_transform.can_handle(csv_path, capital_one_credit_config) is False

    def test_not_handled_by_capital_one_360(self, capital_one_credit_csv_file, capital_one_config):
        """Test that the Capital One 360 transform does not claim the credit card file."""
        assert capital_one_transform.can_handle(capital_one_credit_csv_file, capital_one_config) is False

    def test_load_preserves_all_transactions(self, capital_one_credit_csv_file, capital_one_credit_config):
        """Test all transactions are preserved."""
        result = capital_one_credit_transform.load(capital_one_credit_csv_file, capital_one_credit_config)
        assert len(result) == 4

    def test_load_has_all_target_columns(self, capital_one_credit_csv_file, capital_one_credit_config):
        """Test output has all target columns."""
        result = capital_one_credit_transform.load(capital_one_credit_csv_file, capital_one_credit_config)
        assert all(col in result.columns for col in ['Date', 'Account', 'Amount', 'Subcategory', 'Memo', 'Currency', 'AccountType'])

    def test_load_date_parsing(self, capital_one_credit_csv_file, capital_one_credit_config):
        """Test that dates are parsed from the Transaction Date (YYYY-MM-DD format)."""
        result = capital_one_credit_transform.load(capital_one_credit_csv_file, capital_one_credit_config)
        assert pd.api.types.is_datetime64_any_dtype(result.Date)
        assert result.Date.iloc[0] == pd.Timestamp(2026, 10, 28)

    def test_load_debit_amounts_negative(self, capital_one_credit_csv_file, capital_one_credit_config):
        """Test that Debit amounts (purchases) are negative."""
        result = capital_one_credit_transform.load(capital_one_credit_csv_file, capital_one_credit_config)
        assert result.Amount.iloc[0] == pytest.approx(-1.99)
        assert result.Amount.iloc[1] == pytest.approx(-80.70)

    def test_load_credit_amounts_positive(self, capital_one_credit_csv_file, capital_one_credit_config):
        """Test that Credit amounts (payments and refunds) are positive."""
        result = capital_one_credit_transform.load(capital_one_credit_csv_file, capital_one_credit_config)
        assert result.Amount.iloc[2] == pytest.approx(615.17)
        assert result.Amount.iloc[3] == pytest.approx(26.65)

    def test_load_account_name(self, capital_one_credit_csv_file, capital_one_credit_config):
        """Test that Account is set from config."""
        result = capital_one_credit_transform.load(capital_one_credit_csv_file, capital_one_credit_config)
        assert (result.Account == 'Discover Credit').all()

    def test_load_currency_is_usd(self, capital_one_credit_csv_file, capital_one_credit_config):
        """Test that currency is USD."""
        result = capital_one_credit_transform.load(capital_one_credit_csv_file, capital_one_credit_config)
        assert (result.Currency == 'USD').all()

    def test_load_account_type(self, capital_one_credit_csv_file, capital_one_credit_config):
        """Test that AccountType is set from config."""
        result = capital_one_credit_transform.load(capital_one_credit_csv_file, capital_one_credit_config)
        assert (result.AccountType == 'current').all()

    def test_load_category_preserved(self, capital_one_credit_csv_file, capital_one_credit_config):
        """Test that Category is preserved as Subcategory."""
        result = capital_one_credit_transform.load(capital_one_credit_csv_file, capital_one_credit_config)
        assert result.Subcategory.iloc[0] == 'Internet'
        assert result.Subcategory.iloc[2] == 'Payment/Credit'

    def test_load_memo_from_description(self, capital_one_credit_csv_file, capital_one_credit_config):
        """Test that Memo comes from Description."""
        result = capital_one_credit_transform.load(capital_one_credit_csv_file, capital_one_credit_config)
        assert result.Memo.iloc[1] == 'PAYPAL *DIMSUMSAM'
