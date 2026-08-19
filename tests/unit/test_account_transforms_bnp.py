"""Tests for BNP Cash and BNP Stock transforms."""
import os
import pytest
import pandas as pd
import configparser

from bkanalysis.transforms.account_transforms import bnp_cash_transform, bnp_stock_transform


# ==================== BNP Cash Tests ====================

@pytest.fixture
def bnp_cash_config():
    """Provide BNP Cash transform configuration."""
    config = configparser.ConfigParser()
    config['bnpcash'] = {
        'expected_columns': "['Date operation', 'Sous Categorie operation', 'Libelle operation', 'Montant operation']",
        'account_name': 'BNP Cash Account',
        'currency': 'EUR',
        'account_type': 'Current Account',
    }
    return config['bnpcash']


@pytest.fixture
def bnp_cash_csv_file(tmp_path):
    """Create a sample BNP Cash CSV file."""
    csv_content = """Date operation,Sous Categorie operation,Libelle operation,Montant operation
2024-01-15,Virement,Transfer to savings,-1500.00
2024-01-14,Salaire,Monthly salary,3000.00
2024-01-13,Retrait,ATM withdrawal,-200.00
"""
    csv_path = os.path.join(tmp_path, "bnp_cash.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


class TestBNPCashTransform:
    """Test BNP Cash transform."""

    def test_can_handle_valid_csv(self, bnp_cash_csv_file, bnp_cash_config):
        """Test that can_handle recognizes valid CSV."""
        assert bnp_cash_transform.can_handle(bnp_cash_csv_file, bnp_cash_config)

    def test_can_handle_invalid_columns(self, tmp_path, bnp_cash_config):
        """Test that can_handle rejects files with wrong columns."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("Date,Amount\n2024-01-15,100")
        assert not bnp_cash_transform.can_handle(str(bad_csv), bnp_cash_config)

    def test_load_valid_csv(self, bnp_cash_csv_file, bnp_cash_config):
        """Test loading valid CSV returns proper DataFrame."""
        result = bnp_cash_transform.load(bnp_cash_csv_file, bnp_cash_config)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_load_date_parsing(self, bnp_cash_csv_file, bnp_cash_config):
        """Test that dates are properly parsed."""
        result = bnp_cash_transform.load(bnp_cash_csv_file, bnp_cash_config)
        
        assert result.iloc[0]['Date'] == pd.Timestamp('2024-01-15')

    def test_load_amount_from_montant(self, bnp_cash_csv_file, bnp_cash_config):
        """Test that amount comes from Montant operation."""
        result = bnp_cash_transform.load(bnp_cash_csv_file, bnp_cash_config)
        
        assert result.iloc[0]['Amount'] == -1500.00
        assert result.iloc[1]['Amount'] == 3000.00
        assert result.iloc[2]['Amount'] == -200.00

    def test_load_account_name_set(self, bnp_cash_csv_file, bnp_cash_config):
        """Test that account name is set from config."""
        result = bnp_cash_transform.load(bnp_cash_csv_file, bnp_cash_config)
        
        assert (result['Account'] == 'BNP Cash Account').all()

    def test_load_currency_set(self, bnp_cash_csv_file, bnp_cash_config):
        """Test that currency is set from config."""
        result = bnp_cash_transform.load(bnp_cash_csv_file, bnp_cash_config)
        
        assert (result['Currency'] == 'EUR').all()

    def test_load_account_type_set(self, bnp_cash_csv_file, bnp_cash_config):
        """Test that account type is set from config."""
        result = bnp_cash_transform.load(bnp_cash_csv_file, bnp_cash_config)
        
        assert (result['AccountType'] == 'Current Account').all()

    def test_load_subcategory_preserved(self, bnp_cash_csv_file, bnp_cash_config):
        """Test that Subcategory comes from Sous Categorie operation."""
        result = bnp_cash_transform.load(bnp_cash_csv_file, bnp_cash_config)
        
        assert 'Virement' in result['Subcategory'].tolist()
        assert 'Salaire' in result['Subcategory'].tolist()
        assert 'Retrait' in result['Subcategory'].tolist()

    def test_load_has_all_target_columns(self, bnp_cash_csv_file, bnp_cash_config):
        """Test that output has all required target columns."""
        result = bnp_cash_transform.load(bnp_cash_csv_file, bnp_cash_config)
        
        expected_cols = ['Date', 'Account', 'Currency', 'Amount', 'Memo', 'Subcategory', 'AccountType']
        assert all(col in result.columns for col in expected_cols)


# ==================== BNP Stock Tests ====================

@pytest.fixture
def bnp_stock_config():
    """Provide BNP Stock transform configuration."""
    config = configparser.ConfigParser()
    config['bnpstock'] = {
        'expected_columns': "['Date', 'Libelle', 'Statut', 'Quantite', 'ISIN']",
        'account_name': 'BNP Stock Portfolio',
        'account_type': 'Investment',
    }
    return config['bnpstock']


@pytest.fixture
def bnp_stock_csv_file(tmp_path):
    """Create a sample BNP Stock CSV file."""
    csv_content = """Date,Libelle,Statut,Quantite,ISIN
2024-01-15,Apple Inc.,Active,10,US0378331005
2024-01-14,Nestle SA,Active,25,CH0038863350
2024-01-13,LVMH Moet Hennessy,Sold,-15,FR0000121014
"""
    csv_path = os.path.join(tmp_path, "bnp_stock.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_content)
    return csv_path


class TestBNPStockTransform:
    """Test BNP Stock transform."""

    def test_can_handle_valid_csv(self, bnp_stock_csv_file, bnp_stock_config):
        """Test that can_handle recognizes valid CSV."""
        assert bnp_stock_transform.can_handle(bnp_stock_csv_file, bnp_stock_config)

    def test_can_handle_invalid_columns(self, tmp_path, bnp_stock_config):
        """Test that can_handle rejects files with wrong columns."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("Date,Stock\n2024-01-15,Apple")
        assert not bnp_stock_transform.can_handle(str(bad_csv), bnp_stock_config)

    def test_load_valid_csv(self, bnp_stock_csv_file, bnp_stock_config):
        """Test loading valid CSV returns proper DataFrame."""
        result = bnp_stock_transform.load(bnp_stock_csv_file, bnp_stock_config)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_load_date_parsing(self, bnp_stock_csv_file, bnp_stock_config):
        """Test that dates are properly parsed."""
        result = bnp_stock_transform.load(bnp_stock_csv_file, bnp_stock_config)
        
        assert result.iloc[0]['Date'] == pd.Timestamp('2024-01-15')

    def test_load_quantity_as_amount(self, bnp_stock_csv_file, bnp_stock_config):
        """Test that quantity becomes amount."""
        result = bnp_stock_transform.load(bnp_stock_csv_file, bnp_stock_config)
        
        assert result.iloc[0]['Amount'] == 10
        assert result.iloc[1]['Amount'] == 25
        assert result.iloc[2]['Amount'] == -15

    def test_load_isin_as_currency(self, bnp_stock_csv_file, bnp_stock_config):
        """Test that ISIN becomes currency (ticker proxy)."""
        result = bnp_stock_transform.load(bnp_stock_csv_file, bnp_stock_config)
        
        assert result.iloc[0]['Currency'] == 'US0378331005'
        assert result.iloc[1]['Currency'] == 'CH0038863350'

    def test_load_account_name_set(self, bnp_stock_csv_file, bnp_stock_config):
        """Test that account name is set from config."""
        result = bnp_stock_transform.load(bnp_stock_csv_file, bnp_stock_config)
        
        assert (result['Account'] == 'BNP Stock Portfolio').all()

    def test_load_account_type_set(self, bnp_stock_csv_file, bnp_stock_config):
        """Test that account type is set from config."""
        result = bnp_stock_transform.load(bnp_stock_csv_file, bnp_stock_config)
        
        assert (result['AccountType'] == 'Investment').all()

    def test_load_memo_contains_stock_name(self, bnp_stock_csv_file, bnp_stock_config):
        """Test that memo contains stock name."""
        result = bnp_stock_transform.load(bnp_stock_csv_file, bnp_stock_config)
        
        assert 'Apple Inc.' in result.iloc[0]['Memo']
        assert 'Nestle SA' in result.iloc[1]['Memo']

    def test_load_has_all_target_columns(self, bnp_stock_csv_file, bnp_stock_config):
        """Test that output has all required target columns."""
        result = bnp_stock_transform.load(bnp_stock_csv_file, bnp_stock_config)
        
        expected_cols = ['Date', 'Account', 'Currency', 'Amount', 'Memo', 'Subcategory', 'AccountType']
        assert all(col in result.columns for col in expected_cols)
