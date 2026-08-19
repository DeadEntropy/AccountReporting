"""Tests for Chase Business account transform."""
import os
import pytest
import pandas as pd
import tempfile
import configparser

from bkanalysis.transforms.account_transforms import chasebusiness_transform


@pytest.fixture
def chasebusiness_config():
    """Provide Chase Business transform configuration."""
    config = configparser.ConfigParser()
    config['chasebusiness'] = {
        'expected_columns': "['Posting Date', 'Description', 'Type', 'Amount']",
        'account_name': 'Chase Business',
        'account_type': 'business',
    }
    return config['chasebusiness']


@pytest.fixture
def chasebusiness_csv_file(tmp_path):
    """Create a sample Chase Business CSV file."""
    csv_content = """Posting Date,Description,Type,Amount
01/15/2024,Office Supplies Store,Purchase,-125.50
01/16/2024,Payroll Deposit,Deposit,5000.00
01/17/2024,Fuel Purchase,Purchase,-75.00
01/18/2024,Service Payment,Fee,-50.00
"""
    csv_path = os.path.join(tmp_path, "chase_business_statement.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


@pytest.fixture
def chasebusiness_hyphenated_file(tmp_path):
    """Create a Chase Business CSV with hyphenated filename."""
    csv_content = """Posting Date,Description,Type,Amount
01/15/2024,Office Supplies Store,Purchase,-125.50
01/16/2024,Payroll Deposit,Deposit,5000.00
"""
    # File name: biz-checking-account.csv -> should become "Biz Checking Account"
    csv_path = os.path.join(tmp_path, "biz-checking-account.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


class TestChaseBusinessTransform:
    """Test Chase Business transform."""

    def test_can_handle_valid_csv(self, chasebusiness_csv_file, chasebusiness_config):
        """Test that can_handle recognizes valid Chase Business CSV."""
        assert chasebusiness_transform.can_handle(chasebusiness_csv_file, chasebusiness_config)

    def test_can_handle_invalid_extension(self, tmp_path, chasebusiness_config):
        """Test that can_handle rejects non-CSV files."""
        txt_file = tmp_path / "statement.txt"
        txt_file.write_text("some text")
        assert not chasebusiness_transform.can_handle(str(txt_file), chasebusiness_config)

    def test_can_handle_invalid_columns(self, tmp_path, chasebusiness_config):
        """Test that can_handle rejects files with wrong columns."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("date,amount,category\n2024-01-15,100,food")
        assert not chasebusiness_transform.can_handle(str(bad_csv), chasebusiness_config)

    def test_load_valid_csv(self, chasebusiness_csv_file, chasebusiness_config):
        """Test loading valid Chase Business CSV returns proper DataFrame."""
        result = chasebusiness_transform.load(chasebusiness_csv_file, chasebusiness_config)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert all(col in result.columns for col in ['Date', 'Account', 'Currency', 'Amount', 'Memo', 'Subcategory', 'AccountType'])

    def test_load_date_parsing(self, chasebusiness_csv_file, chasebusiness_config):
        """Test that dates are correctly parsed (MM/DD/YYYY format)."""
        result = chasebusiness_transform.load(chasebusiness_csv_file, chasebusiness_config)
        
        assert pd.api.types.is_datetime64_any_dtype(result['Date'])
        assert result.iloc[0]['Date'] == pd.Timestamp('2024-01-15')

    def test_load_account_name_from_simple_file(self, chasebusiness_csv_file, chasebusiness_config):
        """Test that account name is extracted from simple filename."""
        result = chasebusiness_transform.load(chasebusiness_csv_file, chasebusiness_config)
        
        # File "chase_business_statement.csv" uses underscores, so try_get_account_name will capitalize it
        # The result should be "Chase_business_statement"
        assert len(result) == 4
        assert 'chase' in result.iloc[0]['Account'].lower()

    def test_load_account_name_from_hyphenated(self, chasebusiness_hyphenated_file, chasebusiness_config):
        """Test that account name is extracted from hyphenated filename."""
        result = chasebusiness_transform.load(chasebusiness_hyphenated_file, chasebusiness_config)
        
        # File "biz-checking-account.csv" should become "Biz Checking Account"
        assert (result['Account'] == 'Biz Checking Account').all()

    def test_load_currency_usd(self, chasebusiness_csv_file, chasebusiness_config):
        """Test that currency is always USD."""
        result = chasebusiness_transform.load(chasebusiness_csv_file, chasebusiness_config)
        
        assert (result['Currency'] == 'USD').all()

    def test_load_account_type_set(self, chasebusiness_csv_file, chasebusiness_config):
        """Test that AccountType is set correctly from config."""
        result = chasebusiness_transform.load(chasebusiness_csv_file, chasebusiness_config)
        
        assert (result['AccountType'] == 'business').all()

    def test_load_memo_from_description(self, chasebusiness_csv_file, chasebusiness_config):
        """Test that Memo comes from Description column."""
        result = chasebusiness_transform.load(chasebusiness_csv_file, chasebusiness_config)
        
        assert 'Office Supplies' in result.iloc[0]['Memo']
        assert 'Payroll Deposit' in result.iloc[1]['Memo']

    def test_load_subcategory_from_type(self, chasebusiness_csv_file, chasebusiness_config):
        """Test that Subcategory comes from Type column."""
        result = chasebusiness_transform.load(chasebusiness_csv_file, chasebusiness_config)
        
        assert 'Purchase' in result.iloc[0]['Subcategory']
        assert 'Deposit' in result.iloc[1]['Subcategory']

    def test_load_amounts_preserved(self, chasebusiness_csv_file, chasebusiness_config):
        """Test that amounts are preserved exactly as input."""
        result = chasebusiness_transform.load(chasebusiness_csv_file, chasebusiness_config)
        
        assert float(result.iloc[0]['Amount']) == -125.50
        assert float(result.iloc[1]['Amount']) == 5000.00
        assert float(result.iloc[2]['Amount']) == -75.00
        assert float(result.iloc[3]['Amount']) == -50.00

    def test_load_has_all_target_columns(self, chasebusiness_csv_file, chasebusiness_config):
        """Test that output has all required target columns."""
        result = chasebusiness_transform.load(chasebusiness_csv_file, chasebusiness_config)
        
        expected_cols = ['Date', 'Account', 'Currency', 'Amount', 'Memo', 'Subcategory', 'AccountType']
        assert all(col in result.columns for col in expected_cols)

    def test_try_get_account_name_single_word(self, chasebusiness_config):
        """Test try_get_account_name with single word filename."""
        result = chasebusiness_transform.try_get_account_name("checking", chasebusiness_config)
        assert result == "Checking"

    def test_try_get_account_name_hyphenated(self, chasebusiness_config):
        """Test try_get_account_name with hyphenated filename."""
        result = chasebusiness_transform.try_get_account_name("biz-checking-account", chasebusiness_config)
        assert result == "Biz Checking Account"

    def test_try_get_account_name_capitalization(self, chasebusiness_config):
        """Test try_get_account_name capitalizes each word."""
        result = chasebusiness_transform.try_get_account_name("my-special-account", chasebusiness_config)
        assert result == "My Special Account"

