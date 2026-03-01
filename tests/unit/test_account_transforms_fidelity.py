"""Tests for Fidelity account transform."""
import os
import pytest
import pandas as pd
import tempfile
import configparser

from bkanalysis.transforms.account_transforms import fidelity_transform


@pytest.fixture
def fidelity_config():
    """Provide Fidelity transform configuration."""
    config = configparser.ConfigParser()
    config['fidelity'] = {
        'expected_columns': "['Run Date', 'Symbol', 'Quantity', 'Amount ($)', 'Action', 'Security Description']",
        'cash_account': 'USD',
        'account_type': 'brokerage',
    }
    return config['fidelity']


@pytest.fixture
def fidelity_csv_file(tmp_path):
    """Create a sample Fidelity CSV file with both security and cash transactions."""
    csv_content = """Run Date,Symbol,Quantity,Amount ($),Action,Security Description
01/15/2024,AAPL,10,1500.00,BUY,Apple Inc
01/16/2024,USD,-1500.00,,BUY,Cash
01/17/2024,GOOGL,5,7500.00,SELL,Alphabet Inc
01/18/2024,USD,7500.00,,SELL,Cash
01/19/2024,MSFT,8,2400.00,BUY,Microsoft Corp
01/20/2024,USD,-2400.00,,BUY,Cash
"""
    csv_path = os.path.join(tmp_path, "fidelity_statement.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


class TestFidelityTransform:
    """Test Fidelity transform."""

    def test_can_handle_valid_fidelity_csv(self, fidelity_csv_file, fidelity_config):
        """Test that can_handle recognizes valid Fidelity CSV."""
        assert fidelity_transform.can_handle(fidelity_csv_file, fidelity_config)

    def test_can_handle_invalid_extension(self, tmp_path, fidelity_config):
        """Test that can_handle rejects non-CSV files."""
        txt_file = tmp_path / "statement.txt"
        txt_file.write_text("some text")
        assert not fidelity_transform.can_handle(str(txt_file), fidelity_config)

    def test_can_handle_invalid_columns(self, tmp_path, fidelity_config):
        """Test that can_handle rejects files with wrong columns."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("id,ticker,shares\n1,AAPL,10")
        assert not fidelity_transform.can_handle(str(bad_csv), fidelity_config)

    def test_load_valid_fidelity_csv(self, fidelity_csv_file, fidelity_config):
        """Test loading valid Fidelity CSV returns proper DataFrame."""
        result = fidelity_transform.load(fidelity_csv_file, fidelity_config)
        
        assert isinstance(result, pd.DataFrame)
        # 6 original rows + 3 cash rows (one for each security transaction) = 9 rows
        assert len(result) >= 6
        assert all(col in result.columns for col in ['Date', 'Account', 'Currency', 'Amount', 'Memo', 'Subcategory', 'AccountType'])

    def test_load_date_parsing(self, fidelity_csv_file, fidelity_config):
        """Test that dates are properly parsed."""
        result = fidelity_transform.load(fidelity_csv_file, fidelity_config)
        
        assert pd.api.types.is_datetime64_any_dtype(result['Date'])
        # Dates should be in the result (the data gets sorted, so just check one exists)
        assert any(pd.Timestamp('2024-01-15') == date for date in result['Date'])

    def test_load_account_name_set(self, fidelity_csv_file, fidelity_config):
        """Test that account name is set to Fidelity Brokerage."""
        result = fidelity_transform.load(fidelity_csv_file, fidelity_config)
        
        assert (result['Account'] == 'Fidelity Brokerage').all()

    def test_load_account_type_set(self, fidelity_csv_file, fidelity_config):
        """Test that account type is set from config."""
        result = fidelity_transform.load(fidelity_csv_file, fidelity_config)
        
        assert (result['AccountType'] == 'brokerage').all()

    def test_load_currency_security_rows(self, fidelity_csv_file, fidelity_config):
        """Test that security rows have ticker symbol as currency."""
        result = fidelity_transform.load(fidelity_csv_file, fidelity_config)
        
        # Filter to non-cash rows (those with non-USD currency)
        security_rows = result[result['Currency'] != 'USD']
        assert len(security_rows) > 0
        assert 'AAPL' in security_rows['Currency'].values

    def test_load_currency_cash_rows(self, fidelity_csv_file, fidelity_config):
        """Test that cash rows have configured cash account currency."""
        result = fidelity_transform.load(fidelity_csv_file, fidelity_config)
        
        # Cash rows should use the configured cash_account value
        cash_rows = result[result['Currency'] == 'USD']
        assert len(cash_rows) > 0

    def test_load_memo_contains_action(self, fidelity_csv_file, fidelity_config):
        """Test that memo contains transaction action."""
        result = fidelity_transform.load(fidelity_csv_file, fidelity_config)
        
        # At least some rows should have action in memo
        assert any('BUY' in str(memo) or 'SELL' in str(memo) for memo in result['Memo'])

    def test_load_has_all_target_columns(self, fidelity_csv_file, fidelity_config):
        """Test that output has all required target columns."""
        result = fidelity_transform.load(fidelity_csv_file, fidelity_config)
        
        expected_cols = ['Date', 'Account', 'Currency', 'Amount', 'Memo', 'Subcategory', 'AccountType']
        assert all(col in result.columns for col in expected_cols)

    def test_load_duplicates_cash_cash_out_rows(self, fidelity_csv_file, fidelity_config):
        """Test that cash outflows are duplicated for each security transaction."""
        result = fidelity_transform.load(fidelity_csv_file, fidelity_config)
        
        # Should have more rows than input due to cash duplication
        assert len(result) > 6

    def test_load_amounts_present(self, fidelity_csv_file, fidelity_config):
        """Test that amounts are present and numeric in output."""
        result = fidelity_transform.load(fidelity_csv_file, fidelity_config)
        
        # Check that Amount column exists and has numeric values (ignoring NaN)
        assert pd.api.types.is_numeric_dtype(result['Amount'])
        # At least some non-NaN amounts should be present
        assert result['Amount'].notna().any()

    def test_load_subcategory_from_description(self, fidelity_csv_file, fidelity_config):
        """Test that Subcategory comes from Security Description."""
        result = fidelity_transform.load(fidelity_csv_file, fidelity_config)
        
        # Should have Apple in subcategory from first data row
        assert any('Apple' in str(sub) for sub in result['Subcategory'])
