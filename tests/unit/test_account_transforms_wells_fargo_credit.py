"""Tests for Wells Fargo Credit Card account transform."""
import os
import pytest
import pandas as pd
import configparser
from bkanalysis.transforms.account_transforms import wells_fargo_credit_transform


@pytest.fixture
def wells_fargo_credit_config():
    """Provide Wells Fargo Credit transform configuration."""
    config = configparser.ConfigParser()
    config['wellsfargocredit'] = {
        'expected_columns': "['DATE', 'DESCRIPTION', 'AMOUNT', 'CHECK #', 'STATUS']",
        'currency': 'USD',
        'account_name': 'Wells Fargo Credit',
        'account_type': 'credit',
    }
    return config['wellsfargocredit']


@pytest.fixture
def wells_fargo_credit_csv_file(tmp_path):
    """Create a valid Wells Fargo Credit Card CSV file."""
    csv_content = """"DATE","DESCRIPTION","AMOUNT","CHECK #","STATUS"
"11/15/2026","LOWES #03315* DAVIE FL","-188.69",,"Posted"
"11/13/2026","SQ *MIA NUTRITION Cooper City FL","-5.00",,"Posted"
"11/10/2026","ONLINE PAYMENT THANK YOU","500.00",,"Posted"
"""
    csv_path = os.path.join(tmp_path, "WellFargoCreditCard.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


class TestWellsFargoCreditTransform:
    """Tests for Wells Fargo Credit Card transform."""

    def test_can_handle_valid_csv(self, wells_fargo_credit_csv_file, wells_fargo_credit_config):
        """Test that can_handle recognizes valid Wells Fargo Credit CSV."""
        assert wells_fargo_credit_transform.can_handle(wells_fargo_credit_csv_file, wells_fargo_credit_config) is True

    def test_can_handle_invalid_extension(self, tmp_path, wells_fargo_credit_config):
        """Test that can_handle rejects non-CSV files."""
        txt_file = os.path.join(tmp_path, "statement.txt")
        with open(txt_file, 'w') as f:
            f.write('"DATE","DESCRIPTION","AMOUNT","CHECK #","STATUS"')
        assert wells_fargo_credit_transform.can_handle(txt_file, wells_fargo_credit_config) is False

    def test_can_handle_invalid_columns(self, tmp_path, wells_fargo_credit_config):
        """Test that can_handle rejects CSV with wrong columns."""
        csv_path = os.path.join(tmp_path, "wrong_cols.csv")
        with open(csv_path, 'w') as f:
            f.write("Date,Description,Amount\n11/15/2026,Test,500")
        assert wells_fargo_credit_transform.can_handle(csv_path, wells_fargo_credit_config) is False

    def test_load_preserves_all_transactions(self, wells_fargo_credit_csv_file, wells_fargo_credit_config):
        """Test all transactions are preserved."""
        result = wells_fargo_credit_transform.load(wells_fargo_credit_csv_file, wells_fargo_credit_config)
        assert len(result) == 3

    def test_load_has_all_target_columns(self, wells_fargo_credit_csv_file, wells_fargo_credit_config):
        """Test output has all target columns."""
        result = wells_fargo_credit_transform.load(wells_fargo_credit_csv_file, wells_fargo_credit_config)
        assert all(col in result.columns for col in ['Date', 'Account', 'Amount', 'Subcategory', 'Memo', 'Currency', 'AccountType'])

    def test_load_date_parsing(self, wells_fargo_credit_csv_file, wells_fargo_credit_config):
        """Test that dates are parsed from DATE (MM/DD/YYYY format)."""
        result = wells_fargo_credit_transform.load(wells_fargo_credit_csv_file, wells_fargo_credit_config)
        assert pd.api.types.is_datetime64_any_dtype(result.Date)
        assert result.Date.iloc[0] == pd.Timestamp(2026, 11, 15)

    def test_load_amounts_keep_sign(self, wells_fargo_credit_csv_file, wells_fargo_credit_config):
        """Test that purchases stay negative and payments stay positive."""
        result = wells_fargo_credit_transform.load(wells_fargo_credit_csv_file, wells_fargo_credit_config)
        assert result.Amount.iloc[0] == pytest.approx(-188.69)
        assert result.Amount.iloc[1] == pytest.approx(-5.00)
        assert result.Amount.iloc[2] == pytest.approx(500.00)

    def test_load_account_name(self, wells_fargo_credit_csv_file, wells_fargo_credit_config):
        """Test that Account is set from config."""
        result = wells_fargo_credit_transform.load(wells_fargo_credit_csv_file, wells_fargo_credit_config)
        assert (result.Account == 'Wells Fargo Credit').all()

    def test_load_currency_is_usd(self, wells_fargo_credit_csv_file, wells_fargo_credit_config):
        """Test that currency is USD."""
        result = wells_fargo_credit_transform.load(wells_fargo_credit_csv_file, wells_fargo_credit_config)
        assert (result.Currency == 'USD').all()

    def test_load_account_type(self, wells_fargo_credit_csv_file, wells_fargo_credit_config):
        """Test that AccountType is set from config."""
        result = wells_fargo_credit_transform.load(wells_fargo_credit_csv_file, wells_fargo_credit_config)
        assert (result.AccountType == 'credit').all()

    def test_load_memo_from_description(self, wells_fargo_credit_csv_file, wells_fargo_credit_config):
        """Test that Memo comes from DESCRIPTION."""
        result = wells_fargo_credit_transform.load(wells_fargo_credit_csv_file, wells_fargo_credit_config)
        assert result.Memo.iloc[0] == 'LOWES #03315* DAVIE FL'
