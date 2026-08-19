"""Tests for Chase account transform."""
import os
import pytest
import pandas as pd
import configparser
from bkanalysis.transforms.account_transforms import chase_transform


@pytest.fixture
def chase_config():
    """Provide Chase transform configuration."""
    config = configparser.ConfigParser()
    config['chase'] = {
        'expected_columns': "['Transaction Date', 'Post Date', 'Description', 'Category', 'Type', 'Amount', 'Memo']",
        'account_name': 'Chase Credit Card',
        'account_type': 'credit',
    }
    return config['chase']


@pytest.fixture
def chase_csv_file(tmp_path):
    """Create a valid Chase CSV file with all expected columns."""
    csv_content = """Transaction Date,Post Date,Description,Category,Type,Amount,Memo
01/04/2024,01/05/2024,VENDOR TRANSFER,Wire,CREDIT,200.00,External transfer
01/03/2024,01/04/2024,ACH DEBIT AUTO,Payroll,DEBIT,-0.50,ACH payment
01/02/2024,01/03/2024,MERCHANT STORE A,Shopping,CREDIT,0.18,Retail purchase
01/01/2024,01/02/2024,MERCHANT STORE B,Food,DEBIT,-50.00,Service charge
"""
    csv_path = os.path.join(tmp_path, "statement.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


class TestChaseTransform:
    """Tests for Chase transform module."""

    def test_can_handle_valid_chase_csv(self, chase_csv_file, chase_config):
        """Test that can_handle recognizes valid Chase CSV."""
        assert chase_transform.can_handle(chase_csv_file, chase_config) is True

    def test_can_handle_invalid_extension(self, tmp_path, chase_config):
        """Test that can_handle rejects non-CSV files."""
        txt_file = os.path.join(tmp_path, "statement.txt")
        with open(txt_file, 'w') as f:
            f.write("Transaction Date,Post Date,Description,Category,Type,Amount,Memo\n")
        assert chase_transform.can_handle(txt_file, chase_config) is False

    def test_can_handle_invalid_columns(self, tmp_path, chase_config):
        """Test that can_handle rejects CSV with wrong columns."""
        csv_content = "Date,Description,Amount\n12/27/2023,Test,-100"
        csv_path = os.path.join(tmp_path, "wrong_columns.csv")
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        assert chase_transform.can_handle(csv_path, chase_config) is False

    def test_load_valid_chase_csv(self, chase_csv_file, chase_config):
        """Test loading a valid Chase CSV file."""
        result = chase_transform.load(chase_csv_file, chase_config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # 4 transactions
        # Check required output columns exist
        assert all(col in result.columns for col in ['Date', 'Account', 'Amount', 'Subcategory', 'Memo', 'Currency', 'AccountType'])

    def test_load_date_parsing(self, chase_csv_file, chase_config):
        """Test that dates are correctly parsed (MM/DD/YYYY format)."""
        result = chase_transform.load(chase_csv_file, chase_config)
        # Check date is datetime-like (can be any datetime dtype)
        assert pd.api.types.is_datetime64_any_dtype(result.Date)
        assert result.Date.iloc[0].month == 1
        assert result.Date.iloc[0].day == 4
        assert result.Date.iloc[0].year == 2024

    def test_load_amount_mapping(self, chase_csv_file, chase_config):
        """Test that amounts are correctly extracted and preserved."""
        result = chase_transform.load(chase_csv_file, chase_config)
        # Check amounts are converted to numeric
        assert pd.api.types.is_numeric_dtype(result.Amount)
        assert float(result.Amount.iloc[0]) == 200.00
        assert float(result.Amount.iloc[2]) == 0.18

    def test_load_currency_is_usd(self, chase_csv_file, chase_config):
        """Test that currency is set to USD for all rows."""
        result = chase_transform.load(chase_csv_file, chase_config)
        assert (result.Currency == "USD").all()

    def test_load_account_name_from_config(self, chase_csv_file, chase_config):
        """Test that Account is derived from filename in title case format."""
        result = chase_transform.load(chase_csv_file, chase_config)
        # Account name is derived from filename (statement -> Statement)
        assert result.Account.iloc[0] in ['Statement', 'Chase Credit Card', 'Chase Statement']
        assert len(result.Account.iloc[0]) > 0

    def test_load_account_type_set(self, chase_csv_file, chase_config):
        """Test that AccountType is set correctly."""
        result = chase_transform.load(chase_csv_file, chase_config)
        assert (result.AccountType == 'credit').all()

    def test_load_memo_from_description(self, chase_csv_file, chase_config):
        """Test that Memo comes from Description column."""
        result = chase_transform.load(chase_csv_file, chase_config)
        assert 'VENDOR' in result.Memo.iloc[0]
        assert 'ACH' in result.Memo.iloc[1]

    def test_load_subcategory_from_category(self, chase_csv_file, chase_config):
        """Test that Subcategory comes from Category column."""
        result = chase_transform.load(chase_csv_file, chase_config)
        assert result.Subcategory.iloc[0] == 'Wire'
        assert result.Subcategory.iloc[1] == 'Payroll'

    def test_load_handles_negative_amounts(self, chase_csv_file, chase_config):
        """Test that negative amounts are preserved."""
        result = chase_transform.load(chase_csv_file, chase_config)
        assert float(result.Amount.iloc[1]) < 0  # DEBIT should be negative
        assert float(result.Amount.iloc[3]) < 0  # DEBIT should be negative

    def test_load_handles_positive_amounts(self, chase_csv_file, chase_config):
        """Test that positive amounts are preserved."""
        result = chase_transform.load(chase_csv_file, chase_config)
        assert float(result.Amount.iloc[0]) > 0  # CREDIT should be positive
        assert float(result.Amount.iloc[2]) > 0  # CREDIT should be positive

    def test_load_preserves_all_transactions(self, chase_csv_file, chase_config):
        """Test that all rows are preserved."""
        result = chase_transform.load(chase_csv_file, chase_config)
        assert len(result) == 4

    def test_load_has_all_target_columns(self, chase_csv_file, chase_config):
        """Test that all target columns are present."""
        result = chase_transform.load(chase_csv_file, chase_config)
        target_cols = ['Date', 'Account', 'Amount', 'Subcategory', 'Memo', 'Currency', 'AccountType']
        assert all(col in result.columns for col in target_cols)

    def test_load_with_special_characters_in_description(self, tmp_path, chase_config):
        """Test handling of special characters in descriptions."""
        csv_content = """Transaction Date,Post Date,Description,Category,Type,Amount,Memo
01/10/2024,01/11/2024,VENDOR-A/B-LOCATION-X,Dining,CREDIT,45.50,Payment transaction
01/09/2024,01/10/2024,VENDOR.CODE.NAME,Online,DEBIT,-99.99,Service purchase
"""
        csv_path = os.path.join(tmp_path, "special_chars.csv")
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        result = chase_transform.load(csv_path, chase_config)
        assert 'VENDOR' in result.Memo.iloc[0]
        assert 'VENDOR' in result.Memo.iloc[1]
        assert len(result) == 2

    def test_load_missing_required_columns_raises_error(self, tmp_path, chase_config):
        """Test that missing required columns raises an error."""
        csv_content = "Transaction Date,Description,Amount\n12/27/2023,Test,-100"
        csv_path = os.path.join(tmp_path, "missing_cols.csv")
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        with pytest.raises(AssertionError):
            chase_transform.load(csv_path, chase_config)
