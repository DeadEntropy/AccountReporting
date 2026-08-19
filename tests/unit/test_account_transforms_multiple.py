"""Tests for multiple account transforms: Lloyds, Capital One, Marcus, Citi."""
import os
import pytest
import pandas as pd
import configparser
from bkanalysis.transforms.account_transforms import (
    lloyds_current_transform, capital_one_transform, marcus_transform, citi_transform
)


# ===================== LLOYDS FIXTURES =====================

@pytest.fixture
def lloyds_config():
    """Provide Lloyds Current transform configuration."""
    config = configparser.ConfigParser()
    config['lloydscurrent'] = {
        'expected_columns': "['Transaction Date', 'Transaction Type', 'Sort Code', 'Account Number', 'Transaction Description', 'Debit Amount', 'Credit Amount', 'Balance']",
        'currency': 'GBP',
        'account_type': 'current',
    }
    return config['lloydscurrent']


@pytest.fixture
def lloyds_csv_file(tmp_path):
    """Create a valid Lloyds Current CSV file."""
    csv_content = """Transaction Date,Transaction Type,Sort Code,Account Number,Transaction Description,Debit Amount,Credit Amount,Balance
05/01/2024,Standing Order,12-34-56,98765432,Monthly Payment,14.99,,1000.00
04/01/2024,Income,12-34-56,98765432,Income Deposit,,2500.00,1014.99
03/01/2024,Direct Debit,12-34-56,98765432,Service Payment,85.00,,800.00
02/01/2024,Transfer,12-34-56,98765432,Account Transfer,100.00,,200.00
"""
    csv_path = os.path.join(tmp_path, "current_statement.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


class TestLloydsTransform:
    """Tests for Lloyds Current transform."""

    def test_can_handle_valid_lloyds_csv(self, lloyds_csv_file, lloyds_config):
        """Test that can_handle recognizes valid Lloyds CSV."""
        assert lloyds_current_transform.can_handle(lloyds_csv_file, lloyds_config) is True

    def test_load_valid_lloyds_csv(self, lloyds_csv_file, lloyds_config):
        """Test loading a valid Lloyds CSV."""
        result = lloyds_current_transform.load(lloyds_csv_file, lloyds_config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert all(col in result.columns for col in ['Date', 'Account', 'Amount', 'Subcategory', 'Memo', 'Currency', 'AccountType'])

    def test_load_date_parsing(self, lloyds_csv_file, lloyds_config):
        """Test that dates are correctly parsed (DD/MM/YYYY format)."""
        result = lloyds_current_transform.load(lloyds_csv_file, lloyds_config)
        assert pd.api.types.is_datetime64_any_dtype(result.Date)

    def test_load_paid_out_negative(self, lloyds_csv_file, lloyds_config):
        """Test that Debit Amount becomes negative."""
        result = lloyds_current_transform.load(lloyds_csv_file, lloyds_config)
        # First row: Standing Order, Debit 14.99 -> Amount should be -14.99
        assert float(result.Amount.iloc[0]) < 0

    def test_load_paid_in_positive(self, lloyds_csv_file, lloyds_config):
        """Test that Credit Amount is positive."""
        result = lloyds_current_transform.load(lloyds_csv_file, lloyds_config)
        # Second row: Salary, Credit 2500 -> Amount should be 2500
        assert float(result.Amount.iloc[1]) > 0

    def test_load_currency_gbp(self, lloyds_csv_file, lloyds_config):
        """Test that currency is GBP."""
        result = lloyds_current_transform.load(lloyds_csv_file, lloyds_config)
        assert (result.Currency == 'GBP').all()

    def test_load_memo_from_description(self, lloyds_csv_file, lloyds_config):
        """Test that Memo comes from Transaction Description."""
        result = lloyds_current_transform.load(lloyds_csv_file, lloyds_config)
        assert 'Monthly' in result.Memo.iloc[0]
        assert 'Income' in result.Memo.iloc[1]

    def test_load_account_type_set(self, lloyds_csv_file, lloyds_config):
        """Test that AccountType is set correctly."""
        result = lloyds_current_transform.load(lloyds_csv_file, lloyds_config)
        assert (result.AccountType == 'current').all()


# ===================== CAPITAL ONE FIXTURES =====================

@pytest.fixture
def capital_one_config():
    """Provide Capital One transform configuration."""
    config = configparser.ConfigParser()
    config['capitalone'] = {
        'expected_columns': "['Account Number', 'Transaction Date', 'Transaction Amount', 'Transaction Type', 'Transaction Description', 'Balance']",
        'account_name': 'CapitalOne',
        'account_type': 'liquid',
    }
    return config['capitalone']


@pytest.fixture
def capital_one_csv_file(tmp_path):
    """Create a valid Capital One CSV file."""
    # Capital One uses %m/%d/%y format (2-digit year)
    csv_content = """Account Number,Transaction Date,Transaction Amount,Transaction Type,Transaction Description,Balance
111111111,01/05/24,1500.00,Deposit,Transfer In,5000.00
111111111,01/04/24,-50.00,Withdrawal,Cash Withdrawal,3500.00
111111111,01/03/24,100.00,Deposit,Deposit In,3550.00
111111111,01/02/24,-25.50,Fee,Account Fee,3450.00
"""
    csv_path = os.path.join(tmp_path, "savings_statement.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


class TestCapitalOneTransform:
    """Tests for Capital One transform."""

    def test_can_handle_valid_capital_one_csv(self, capital_one_csv_file, capital_one_config):
        """Test that can_handle recognizes valid Capital One CSV."""
        assert capital_one_transform.can_handle(capital_one_csv_file, capital_one_config) is True

    def test_load_valid_capital_one_csv(self, capital_one_csv_file, capital_one_config):
        """Test loading a valid Capital One CSV."""
        result = capital_one_transform.load(capital_one_csv_file, capital_one_config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert all(col in result.columns for col in ['Date', 'Account', 'Amount', 'Subcategory', 'Memo', 'Currency', 'AccountType'])

    def test_load_date_parsing(self, capital_one_csv_file, capital_one_config):
        """Test that dates are correctly parsed (MM/DD/YY format)."""
        result = capital_one_transform.load(capital_one_csv_file, capital_one_config)
        assert pd.api.types.is_datetime64_any_dtype(result.Date)

    def test_load_withdrawal_negative(self, capital_one_csv_file, capital_one_config):
        """Test that withdrawals are negative."""
        result = capital_one_transform.load(capital_one_csv_file, capital_one_config)
        assert float(result.Amount.iloc[1]) < 0  # Second row is a withdrawal

    def test_load_deposit_positive(self, capital_one_csv_file, capital_one_config):
        """Test that deposits are positive."""
        result = capital_one_transform.load(capital_one_csv_file, capital_one_config)
        assert float(result.Amount.iloc[0]) > 0  # First row is a deposit

    def test_load_currency_usd(self, capital_one_csv_file, capital_one_config):
        """Test that currency is USD."""
        result = capital_one_transform.load(capital_one_csv_file, capital_one_config)
        assert (result.Currency == 'USD').all()

    def test_load_account_type_set(self, capital_one_csv_file, capital_one_config):
        """Test that AccountType is set correctly."""
        result = capital_one_transform.load(capital_one_csv_file, capital_one_config)
        assert (result.AccountType == 'liquid').all()

    def test_load_memo_from_description(self, capital_one_csv_file, capital_one_config):
        """Test that Memo comes from Transaction Description."""
        result = capital_one_transform.load(capital_one_csv_file, capital_one_config)
        assert 'Transfer' in result.Memo.iloc[0]


# ===================== MARCUS FIXTURES =====================

@pytest.fixture
def marcus_config():
    """Provide Marcus transform configuration."""
    config = configparser.ConfigParser()
    config['marcus'] = {
        'expected_columns': "['Date', 'Account', 'Amount', 'Subcategory', 'Memo']",
        'currency': 'USD',
        'account_type': 'saving',
    }
    return config['marcus']


@pytest.fixture
def marcus_csv_file(tmp_path):
    """Create a valid Marcus CSV file."""
    # Marcus uses already-formatted columns matching the target format
    csv_content = """Date,Account,Amount,Subcategory,Memo
01/05/2024,Account1,250.00,Interest,Interest payment
01/04/2024,Account1,-100.00,Withdrawal,Withdrawal transaction
01/03/2024,Account1,1000.00,Deposit,Deposit transaction
01/02/2024,Account1,50.00,Interest,Interest earned
"""
    csv_path = os.path.join(tmp_path, "savings_account.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


class TestMarcusTransform:
    """Tests for Marcus transform."""

    def test_can_handle_valid_marcus_csv(self, marcus_csv_file, marcus_config):
        """Test that can_handle recognizes valid Marcus CSV."""
        assert marcus_transform.can_handle(marcus_csv_file, marcus_config) is True

    def test_load_valid_marcus_csv(self, marcus_csv_file, marcus_config):
        """Test loading a valid Marcus CSV."""
        result = marcus_transform.load(marcus_csv_file, marcus_config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert all(col in result.columns for col in ['Date', 'Account', 'Amount', 'Subcategory', 'Memo', 'Currency', 'AccountType'])

    def test_load_currency_usd(self, marcus_csv_file, marcus_config):
        """Test that currency is USD."""
        result = marcus_transform.load(marcus_csv_file, marcus_config)
        assert (result.Currency == 'USD').all()

    def test_load_account_type_set(self, marcus_csv_file, marcus_config):
        """Test that AccountType is set correctly."""
        result = marcus_transform.load(marcus_csv_file, marcus_config)
        assert (result.AccountType == 'saving').all()

    def test_load_preserves_amounts(self, marcus_csv_file, marcus_config):
        """Test that amounts are preserved."""
        result = marcus_transform.load(marcus_csv_file, marcus_config)
        assert float(result.Amount.iloc[0]) == 250.00
        assert float(result.Amount.iloc[1]) == -100.00


# ===================== CITI FIXTURES =====================

@pytest.fixture
def citi_config():
    """Provide Citi transform configuration."""
    config = configparser.ConfigParser()
    config['citi'] = {
        'expected_columns': "['Unnamed: 0', 'Date', 'Description', 'Debit', 'Credit']",
        'currency': 'USD',
        'account_type': 'current',
    }
    return config['citi']


@pytest.fixture
def citi_csv_file(tmp_path):
    """Create a valid Citi CSV file."""
    # Citi format uses MM-DD-YYYY and includes an unnamed first column
    csv_content = """,Date,Description,Debit,Credit
1,01-05-2024,Income Deposit,,2500.00
2,01-04-2024,Withdrawal,100.00,
3,01-03-2024,Transfer In,,500.00
4,01-02-2024,Payment,150.00,
"""
    csv_path = os.path.join(tmp_path, "account_statement.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


class TestCitiTransform:
    """Tests for Citi transform."""

    def test_can_handle_valid_citi_csv(self, citi_csv_file, citi_config):
        """Test that can_handle recognizes valid Citi CSV."""
        assert citi_transform.can_handle(citi_csv_file, citi_config) is True

    def test_load_valid_citi_csv(self, citi_csv_file, citi_config):
        """Test loading a valid Citi CSV."""
        result = citi_transform.load(citi_csv_file, citi_config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert all(col in result.columns for col in ['Date', 'Account', 'Amount', 'Subcategory', 'Memo', 'Currency', 'AccountType'])

    def test_load_date_parsing(self, citi_csv_file, citi_config):
        """Test that dates are correctly parsed (MM-DD-YYYY format)."""
        result = citi_transform.load(citi_csv_file, citi_config)
        assert pd.api.types.is_datetime64_any_dtype(result.Date)

    def test_load_currency_mapping(self, citi_csv_file, citi_config):
        """Test that currency is USD."""
        result = citi_transform.load(citi_csv_file, citi_config)
        assert (result.Currency == 'USD').all()

    def test_load_amount_handling(self, citi_csv_file, citi_config):
        """Test that amounts are correctly calculated (Credit - Debit)."""
        result = citi_transform.load(citi_csv_file, citi_config)
        # First row: Credit 2500, Debit 0 -> Amount = 2500
        assert float(result.Amount.iloc[0]) == 2500.00
        # Second row: Debit 100, Credit 0 -> Amount = -100
        assert float(result.Amount.iloc[1]) == -100.00
