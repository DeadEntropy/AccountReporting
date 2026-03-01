"""Tests for First Republic account transform."""
import os
import pytest
import pandas as pd
import tempfile
import configparser

from bkanalysis.transforms.account_transforms import first_republic_transform


@pytest.fixture
def first_republic_config():
    """Provide First Republic transform configuration."""
    config = configparser.ConfigParser()
    config['firstrepublic'] = {
        'expected_columns': "['Date', 'Category', 'Description', 'Debit', 'Credit']",
        'account_type': 'checking',
    }
    return config['firstrepublic']


@pytest.fixture
def first_republic_csv_file(tmp_path):
    """Create a sample First Republic CSV file."""
    csv_content = """Date,Category,Description,Debit,Credit
01/15/2024,Deposit,Salary,,5000.00
01/16/2024,Withdrawal,ATM Cash,500.00,
01/17/2024,Mobile Deposit,Mobile Deposit,1200.00,
01/18/2024,Transfer,Online Transfer,250.00,
01/19/2024,Deposit,Wire Transfer,,2500.00
"""
    csv_path = os.path.join(tmp_path, "first_republic_statement.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


@pytest.fixture
def first_republic_with_empty_fields(tmp_path):
    """Create a First Republic CSV with empty fields."""
    csv_content = """Date,Category,Description,Debit,Credit
01/15/2024,,Empty Description,100.00,
01/16/2024,Some Category,,0,500.00
"""
    csv_path = os.path.join(tmp_path, "first_republic_empty.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


class TestFirstRepublicTransform:
    """Test First Republic transform."""

    def test_can_handle_valid_csv(self, first_republic_csv_file, first_republic_config):
        """Test that can_handle recognizes valid First Republic CSV."""
        assert first_republic_transform.can_handle(first_republic_csv_file, first_republic_config)

    def test_can_handle_invalid_extension(self, tmp_path, first_republic_config):
        """Test that can_handle rejects non-CSV files."""
        txt_file = tmp_path / "statement.txt"
        txt_file.write_text("some text")
        assert not first_republic_transform.can_handle(str(txt_file), first_republic_config)

    def test_can_handle_invalid_columns(self, tmp_path, first_republic_config):
        """Test that can_handle rejects files with wrong columns."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("date,amount\n2024-01-15,100")
        assert not first_republic_transform.can_handle(str(bad_csv), first_republic_config)

    def test_load_valid_csv(self, first_republic_csv_file, first_republic_config):
        """Test loading valid First Republic CSV returns proper DataFrame."""
        result = first_republic_transform.load(first_republic_csv_file, first_republic_config)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert all(col in result.columns for col in ['Date', 'Account', 'Currency', 'Amount', 'Memo', 'Subcategory', 'AccountType'])

    def test_load_date_parsing(self, first_republic_csv_file, first_republic_config):
        """Test that dates are correctly parsed (MM/DD/YYYY format)."""
        result = first_republic_transform.load(first_republic_csv_file, first_republic_config)
        
        assert pd.api.types.is_datetime64_any_dtype(result['Date'])
        assert result.iloc[0]['Date'] == pd.Timestamp('2024-01-15')

    def test_load_account_name(self, first_republic_csv_file, first_republic_config):
        """Test that account name is set to First Republic."""
        result = first_republic_transform.load(first_republic_csv_file, first_republic_config)
        
        assert (result['Account'] == 'First Republic').all()

    def test_load_currency_usd(self, first_republic_csv_file, first_republic_config):
        """Test that currency is always USD."""
        result = first_republic_transform.load(first_republic_csv_file, first_republic_config)
        
        assert (result['Currency'] == 'USD').all()

    def test_load_account_type_set(self, first_republic_csv_file, first_republic_config):
        """Test that AccountType is set from config."""
        result = first_republic_transform.load(first_republic_csv_file, first_republic_config)
        
        assert (result['AccountType'] == 'checking').all()

    def test_load_debit_positive(self, first_republic_csv_file, first_republic_config):
        """Test that Debit amounts are preserved."""
        result = first_republic_transform.load(first_republic_csv_file, first_republic_config)
        
        # Row with 500 debit -> 500 (debit + credit)
        assert result.iloc[1]['Amount'] == 500.00

    def test_load_credit_positive(self, first_republic_csv_file, first_republic_config):
        """Test that Credit amounts are positive."""
        result = first_republic_transform.load(first_republic_csv_file, first_republic_config)
        
        # Row with 5000 credit should be 5000
        assert result.iloc[0]['Amount'] == 5000.00

    def test_load_both_debit_and_credit_empty(self, first_republic_with_empty_fields, first_republic_config):
        """Test handling of empty debit/credit fields."""
        result = first_republic_transform.load(first_republic_with_empty_fields, first_republic_config)
        
        assert len(result) == 2
        # First row: 100 debit, no credit -> 100 + 0 = 100
        assert result.iloc[0]['Amount'] == 100.00
        # Second row: no debit, 500 credit -> 0 + 500 = 500
        assert result.iloc[1]['Amount'] == 500.00

    def test_load_subcategory_from_category(self, first_republic_csv_file, first_republic_config):
        """Test that Subcategory comes from Category column."""
        result = first_republic_transform.load(first_republic_csv_file, first_republic_config)
        
        assert result.iloc[0]['Subcategory'] == 'Deposit'
        assert result.iloc[1]['Subcategory'] == 'Withdrawal'
        assert result.iloc[2]['Subcategory'] == 'Mobile Deposit'

    def test_load_memo_postprocessing_mobile_deposit(self, first_republic_csv_file, first_republic_config):
        """Test that Mobile Deposit memo is augmented with date and amount."""
        result = first_republic_transform.load(first_republic_csv_file, first_republic_config)
        
        # Mobile Deposit row should have memo with date and amount
        mobile_deposit_memo = result.iloc[2]['Memo']
        assert 'Mobile Deposit' in mobile_deposit_memo
        # Should contain formatted date and amount
        assert '2024' in mobile_deposit_memo or '20240117' in mobile_deposit_memo

    def test_load_memo_normal_description_unchanged(self, first_republic_csv_file, first_republic_config):
        """Test that non-Mobile Deposit memos are unchanged."""
        result = first_republic_transform.load(first_republic_csv_file, first_republic_config)
        
        # Regular descriptions should match original
        assert result.iloc[0]['Memo'] == 'Salary'
        assert result.iloc[1]['Memo'] == 'ATM Cash'

    def test_load_handles_empty_description(self, first_republic_with_empty_fields, first_republic_config):
        """Test that empty descriptions are replaced with default."""
        result = first_republic_transform.load(first_republic_with_empty_fields, first_republic_config)
        
        assert result.iloc[0]['Memo'] == 'Empty Description'  # Falls back to Description column
        assert result.iloc[1]['Memo'] == 'Empty Description'  # Falls back to default

    def test_load_has_all_target_columns(self, first_republic_csv_file, first_republic_config):
        """Test that output has all required target columns."""
        result = first_republic_transform.load(first_republic_csv_file, first_republic_config)
        
        expected_cols = ['Date', 'Account', 'Currency', 'Amount', 'Memo', 'Subcategory', 'AccountType']
        assert all(col in result.columns for col in expected_cols)

    def test_load_preserves_all_transactions(self, first_republic_csv_file, first_republic_config):
        """Test that all transactions are preserved."""
        result = first_republic_transform.load(first_republic_csv_file, first_republic_config)
        
        assert len(result) == 5
        # Verify sum: 5000 + 500 + 1200 + 250 + 2500 = 9450 (Debit + Credit)
        assert result['Amount'].sum() == 9450.00
