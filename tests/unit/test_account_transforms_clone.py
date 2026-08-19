"""Tests for Clone account transform."""
import os
import pytest
import pandas as pd

from bkanalysis.transforms.account_transforms import clone_transform


@pytest.fixture
def clone_csv_file_no_account_type(tmp_path):
    """Create a CSV file in target format without AccountType."""
    csv_content = """Date,Account,Currency,Amount,Memo,Subcategory
15/01/2024,Investment Account,USD,1500.00,Dividend Payment,Income
16/01/2024,Investment Account,USD,-500.00,Fee,Charges
17/01/2024,Investment Account,EUR,2000.00,Transfer In,Transfer
"""
    csv_path = os.path.join(tmp_path, "clone_data.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


@pytest.fixture
def clone_csv_file_with_account_type(tmp_path):
    """Create a CSV file in target format with AccountType."""
    csv_content = """Date,Account,Currency,Amount,Memo,Subcategory,AccountType
15/01/2024,Investment Account,USD,1500.00,Dividend Payment,Income,brokerage
16/01/2024,Investment Account,USD,-500.00,Fee,Charges,brokerage
17/01/2024,Investment Account,EUR,2000.00,Transfer In,Transfer,brokerage
"""
    csv_path = os.path.join(tmp_path, "clone_with_type.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


class TestCloneTransform:
    """Test Clone transform."""

    def test_can_handle_valid_target_format_no_account_type(self, clone_csv_file_no_account_type):
        """Test that can_handle recognizes target format without AccountType."""
        assert clone_transform.can_handle(clone_csv_file_no_account_type)

    def test_can_handle_valid_target_format_with_account_type(self, clone_csv_file_with_account_type):
        """Test that can_handle recognizes target format with AccountType."""
        assert clone_transform.can_handle(clone_csv_file_with_account_type)

    def test_can_handle_invalid_extension(self, tmp_path):
        """Test that can_handle rejects non-CSV files."""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("Date,Account,Amount\n01/15/2024,Acc,100")
        assert not clone_transform.can_handle(str(txt_file))

    def test_can_handle_invalid_columns(self, tmp_path):
        """Test that can_handle rejects wrong column set."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("id,date,amount\n1,2024-01-15,100")
        assert not clone_transform.can_handle(str(bad_csv))

    def test_load_no_account_type(self, clone_csv_file_no_account_type):
        """Test loading CSV without AccountType adds default 'flat'."""
        result = clone_transform.load(clone_csv_file_no_account_type)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'AccountType' in result.columns
        assert (result['AccountType'] == 'flat').all()

    def test_load_with_account_type(self, clone_csv_file_with_account_type):
        """Test loading CSV with AccountType preserves it."""
        result = clone_transform.load(clone_csv_file_with_account_type)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'AccountType' in result.columns
        assert (result['AccountType'] == 'brokerage').all()

    def test_load_date_parsing(self, clone_csv_file_no_account_type):
        """Test that dates are properly parsed (DD/MM/YYYY format)."""
        result = clone_transform.load(clone_csv_file_no_account_type)
        
        assert pd.api.types.is_datetime64_any_dtype(result['Date'])

    def test_load_has_all_target_columns(self, clone_csv_file_no_account_type):
        """Test that output has all required target columns."""
        result = clone_transform.load(clone_csv_file_no_account_type)
        
        expected_cols = ['Date', 'Account', 'Currency', 'Amount', 'Memo', 'Subcategory', 'AccountType']
        assert all(col in result.columns for col in expected_cols)

    def test_load_preserves_all_data(self, clone_csv_file_no_account_type):
        """Test that all data is preserved."""
        result = clone_transform.load(clone_csv_file_no_account_type)
        
        assert result.iloc[0]['Account'] == 'Investment Account'
        assert result.iloc[0]['Currency'] == 'USD'
        assert result.iloc[0]['Memo'] == 'Dividend Payment'
        assert result.iloc[0]['Subcategory'] == 'Income'

    def test_load_preserves_currencies(self, clone_csv_file_no_account_type):
        """Test that multiple currencies are preserved."""
        result = clone_transform.load(clone_csv_file_no_account_type)
        
        currencies = result['Currency'].unique()
        assert 'USD' in currencies
        assert 'EUR' in currencies

    def test_load_preserves_amounts(self, clone_csv_file_no_account_type):
        """Test that positive and negative amounts are preserved."""
        result = clone_transform.load(clone_csv_file_no_account_type)
        
        assert (result['Amount'] > 0).any()
        assert (result['Amount'] < 0).any()

    def test_load_returns_dataframe(self, clone_csv_file_with_account_type):
        """Test that load returns a DataFrame."""
        result = clone_transform.load(clone_csv_file_with_account_type)
        assert isinstance(result, pd.DataFrame)

    def test_load_no_modification_with_account_type(self, clone_csv_file_with_account_type):
        """Test that data integrity is maintained when AccountType present."""
        result = clone_transform.load(clone_csv_file_with_account_type)
        
        # All rows should have data
        assert len(result) == 3
        assert not result.isnull().any().any()  # No null values
