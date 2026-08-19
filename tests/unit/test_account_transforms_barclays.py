"""Tests for Barclays account transform."""
import os
import pytest
import pandas as pd
import tempfile
import shutil
from datetime import datetime

from bkanalysis.transforms.account_transforms import barclays_transform
import configparser


@pytest.fixture
def barclays_config():
    """Provide Barclays transform configuration."""
    config = configparser.ConfigParser()
    config['barclays'] = {
        'expected_columns': "['Number', 'Date', 'Account', 'Amount', 'Subcategory', 'Memo']",
        'account_currencies': "{'AA-BB-CC 11111111': 'GBP', 'AA-BB-CC 22222222': 'EUR'}",
        'account_type': 'Barclays Current Account',
    }
    return config['barclays']


@pytest.fixture
def temp_barclays_dir():
    """Create temporary directory for Barclays CSV files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def barclays_csv_file(temp_barclays_dir):
    """Create a sample Barclays CSV file."""
    csv_content = """Number,Date,Account,Amount,Subcategory,Memo
,01/04/2024,AA-BB-CC 11111111,-33,Payment,Service payment
,01/04/2024,AA-BB-CC 11111111,-6.26,Transportation,Service charge
,01/03/2024,AA-BB-CC 11111111,500,Standing Order,Income deposit
,01/02/2024,AA-BB-CC 11111111,-45.99,Merchant,Retail purchase
"""
    csv_path = os.path.join(temp_barclays_dir, "statement.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


@pytest.fixture
def barclays_eur_csv_file(temp_barclays_dir):
    """Create a sample Barclays EUR CSV file."""
    csv_content = """Number,Date,Account,Amount,Subcategory,Memo
,01/05/2024,AA-BB-CC 22222222,-50.5,Travel,Travel expense
,01/04/2024,AA-BB-CC 22222222,1000,Transfer In,Transfer received
"""
    csv_path = os.path.join(temp_barclays_dir, "statement_eur.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


class TestBarclaysTransform:
    """Tests for Barclays transform module."""

    def test_can_handle_valid_barclays_csv(self, barclays_csv_file, barclays_config):
        """Test that can_handle recognizes valid Barclays CSV."""
        assert barclays_transform.can_handle(barclays_csv_file, barclays_config) is True

    def test_can_handle_invalid_columns(self, temp_barclays_dir, barclays_config):
        """Test that can_handle rejects file with wrong columns."""
        csv_path = os.path.join(temp_barclays_dir, "invalid.csv")
        csv_content = "WrongCol1,WrongCol2\nvalue1,value2\n"
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        
        assert barclays_transform.can_handle(csv_path, barclays_config) is False

    def test_can_handle_non_csv(self, temp_barclays_dir):
        """Test that can_handle rejects non-CSV files."""
        txt_path = os.path.join(temp_barclays_dir, "test.txt")
        with open(txt_path, 'w') as f:
            f.write("not a csv")
        
        assert barclays_transform.can_handle(txt_path, {}) is False

    def test_load_valid_barclays_csv(self, barclays_csv_file, barclays_config):
        """Test loading valid Barclays CSV returns proper DataFrame."""
        result = barclays_transform.load(barclays_csv_file, barclays_config)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert all(col in result.columns for col in ['Date', 'Account', 'Amount', 'Subcategory', 'Memo', 'Currency'])
        
        # Verify data
        assert result.iloc[0]['Date'] == pd.Timestamp('2024-04-01')
        assert result.iloc[0]['Amount'] == -33
        assert result.iloc[0]['Currency'] == 'GBP'
        assert 'Number' not in result.columns  # Should be dropped

    def test_load_date_parsing(self, barclays_csv_file, barclays_config):
        """Test that dates are correctly parsed."""
        result = barclays_transform.load(barclays_csv_file, barclays_config)
        
        # Check date column is datetime
        assert pd.api.types.is_datetime64_any_dtype(result['Date'])
        assert result['Date'].iloc[0] == pd.Timestamp('2024-04-01')

    def test_load_currency_mapping(self, barclays_csv_file, barclays_config):
        """Test that currencies are correctly mapped by account."""
        result = barclays_transform.load(barclays_csv_file, barclays_config)
        
        # All rows in this file are from GBP account
        assert (result['Currency'] == 'GBP').all()

    def test_load_memo_cleanup(self, barclays_csv_file, barclays_config):
        """Test that multiple spaces in memo are cleaned up."""
        result = barclays_transform.load(barclays_csv_file, barclays_config)
        
        # Check that multiple spaces are reduced
        for memo in result['Memo']:
            assert '  ' not in memo  # No double spaces

    def test_load_multiple_currencies(self, barclays_csv_file, barclays_eur_csv_file, barclays_config):
        """Test loading file with EUR account."""
        result = barclays_transform.load(barclays_eur_csv_file, barclays_config)
        
        assert len(result) == 2
        assert all(result['Currency'] == 'EUR')
        assert all(result['Account'] == 'AA-BB-CC 22222222')

    def test_load_missing_columns_raises_error(self, temp_barclays_dir, barclays_config):
        """Test that missing expected columns raise AssertionError."""
        csv_path = os.path.join(temp_barclays_dir, "incomplete.csv")
        csv_content = "Number,Date,Account\n,01/01/2024,AA-BB-CC 11111111\n"
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        
        with pytest.raises(AssertionError, match="Was expecting.*but file columns"):
            barclays_transform.load(csv_path, barclays_config)

    def test_load_preserves_amount_precision(self, barclays_csv_file, barclays_config):
        """Test that amount values are preserved with correct precision."""
        result = barclays_transform.load(barclays_csv_file, barclays_config)
        
        assert result.iloc[1]['Amount'] == -6.26
        assert result.iloc[2]['Amount'] == 500

    def test_load_handles_negative_amounts(self, barclays_csv_file, barclays_config):
        """Test that negative amounts are correctly preserved."""
        result = barclays_transform.load(barclays_csv_file, barclays_config)
        
        # Check negative values exist
        negative_amounts = result[result['Amount'] < 0]
        assert len(negative_amounts) > 0
        assert (negative_amounts['Amount'] < 0).all()

    def test_load_account_type_mapping(self, barclays_csv_file, barclays_config):
        """Test that AccountType is correctly set."""
        result = barclays_transform.load(barclays_csv_file, barclays_config)
        
        assert (result['AccountType'] == 'Barclays Current Account').all()

    def test_load_subcategory_preserved(self, barclays_csv_file, barclays_config):
        """Test that Subcategory column is preserved from input."""
        result = barclays_transform.load(barclays_csv_file, barclays_config)
        
        assert result.iloc[0]['Subcategory'] == 'Payment'
        assert result.iloc[2]['Subcategory'] == 'Standing Order'

    def test_load_has_all_target_columns(self, barclays_csv_file, barclays_config):
        """Test that output has all required target columns."""
        result = barclays_transform.load(barclays_csv_file, barclays_config)
        
        expected_cols = {'Date', 'Account', 'Amount', 'Subcategory', 'Memo', 'Currency', 'AccountType'}
        assert expected_cols.issubset(set(result.columns))
