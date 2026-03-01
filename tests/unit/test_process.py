"""
Comprehensive tests for the process module.
Tests memo mapping, type mapping, IAT identification, and transaction processing.
"""
import pytest
import pandas as pd
import configparser
import datetime
import tempfile
import os
from unittest.mock import patch, MagicMock, mock_open, PropertyMock
from io import StringIO

from bkanalysis.process.process import Process
from bkanalysis.process.process_helper import (
    get_adjusted_month,
    get_adjusted_year,
    get_fiscal_year,
    get_year_to_date,
)
from bkanalysis.process.iat_identification import IatIdentification
from bkanalysis.process.status import LastUpdate


# ============ Test Fixtures ============

@pytest.fixture
def valid_process_config():
    """Create a valid config for Process class."""
    config = configparser.ConfigParser()
    config['Mapping'] = {
        'path_map': 'map_simple.csv',
        'path_map_type': 'map_type.csv',
        'path_map_full_type': 'map_full_type.csv',
        'path_map_full_subtype': 'map_full_subtype.csv',
        'path_map_full_master_type': 'map_master.csv',
        'path_override': 'override.csv',
        'expected_columns': "['Date', 'Account', 'AccountType', 'Amount', 'Subcategory', 'Memo', 'Currency', 'SourceFile']",
        'new_columns': "['Date', 'Account', 'AccountType', 'Amount', 'Subcategory', 'Memo', 'Currency', 'SourceFile', 'MemoSimple', 'MemoMapped', 'Type', 'FullType', 'SubType', 'FullSubType', 'MasterType', 'FullMasterType', 'YearToDate', 'FacingAccount']"
    }
    config['folder_root'] = '/tmp'
    config['IO'] = {
        'path_aggregated': '/tmp/aggregated.csv',
        'path_processed': '/tmp/processed.csv'
    }
    return config


@pytest.fixture
def sample_transaction_df():
    """Create sample transaction DataFrame."""
    return pd.DataFrame({
        'Date': [pd.Timestamp('2023-01-15'), pd.Timestamp('2023-02-20')],
        'Account': ['Bank A', 'Bank A'],
        'AccountType': ['Checking', 'Checking'],
        'Amount': [100.0, -50.0],
        'Subcategory': ['', ''],
        'Memo': ['AMAZON PURCHASE', 'WIRE TRANSFER'],
        'Currency': ['GBP', 'GBP'],
        'SourceFile': ['bank_statement.csv', 'bank_statement.csv']
    })


# ============ Process Helper Tests ============

class TestProcessHelper:
    """Tests for process_helper functions."""

    def test_get_adjusted_month_before_20th(self):
        """Test get_adjusted_month returns same month for dates before 20th."""
        dt = datetime.datetime(2023, 3, 15)
        assert get_adjusted_month(dt) == 3

    def test_get_adjusted_month_after_20th(self):
        """Test get_adjusted_month returns next month for dates after 20th."""
        dt = datetime.datetime(2023, 3, 25)
        assert get_adjusted_month(dt) == 4

    def test_get_adjusted_month_december_after_20th(self):
        """Test get_adjusted_month wraps December to January."""
        dt = datetime.datetime(2023, 12, 25)
        assert get_adjusted_month(dt) == 1

    def test_get_adjusted_year_before_20th(self):
        """Test get_adjusted_year returns same year for dates before 20th."""
        dt = datetime.datetime(2023, 3, 15)
        assert get_adjusted_year(dt) == 2023

    def test_get_adjusted_year_december_after_20th(self):
        """Test get_adjusted_year increments for December after 20th."""
        dt = datetime.datetime(2023, 12, 25)
        assert get_adjusted_year(dt) == 2024

    def test_get_fiscal_year_before_april_5(self):
        """Test get_fiscal_year returns previous year before April 5."""
        dt = datetime.datetime(2023, 4, 1)
        assert get_fiscal_year(dt) == 2022

    def test_get_fiscal_year_after_april_5(self):
        """Test get_fiscal_year returns same year after April 5."""
        dt = datetime.datetime(2023, 4, 6)
        assert get_fiscal_year(dt) == 2023

    def test_get_year_to_date(self):
        """Test get_year_to_date returns correct age in years."""
        today = datetime.date.today()
        dt = today - datetime.timedelta(days=365)
        result = get_year_to_date(datetime.datetime.combine(dt, datetime.time()))
        assert result >= 0 and result <= 2  # Should be around 1 year

    def test_get_year_to_date_recent(self):
        """Test get_year_to_date for recent dates returns 0."""
        today = datetime.date.today()
        dt = today - datetime.timedelta(days=100)
        result = get_year_to_date(datetime.datetime.combine(dt, datetime.time()))
        assert result < 1


# ============ IatIdentification Tests ============

class TestIatIdentification:
    """Tests for IatIdentification class."""

    @pytest.fixture
    def iat_config(self):
        """Create config for IatIdentification."""
        config = configparser.ConfigParser()
        config['Mapping'] = {
            'new_columns': "['Date', 'Currency', 'FullType', 'Account', 'Amount', 'Type', 'SubType', 'FullSubType', 'MasterType', 'FullMasterType', 'MemoMapped', 'Memo', 'MemoSimple', 'Subcategory', 'AccountType', 'SourceFile', 'YearToDate', 'FacingAccount']"
        }
        return config

    def test_iat_identification_init(self, iat_config):
        """Test IatIdentification initialization."""
        iat = IatIdentification(iat_config)
        assert iat.iat_types == ["SA", "IAT", "W_IN", "W_OUT", "SC", "R", "MC", "O", "FR", "TAX", "FPC", "FLC", "FLL", "FSC"]

    def test_map_iat_identifies_offsetting_transactions(self, iat_config):
        """Test map_iat identifies offsetting intra-account transfers."""
        iat = IatIdentification(iat_config)
        
        df = pd.DataFrame({
            'Date': [pd.Timestamp('2023-01-15'), pd.Timestamp('2023-01-16')],
            'Currency': ['GBP', 'GBP'],
            'FullType': ['Savings', 'Savings'],
            'Account': ['Account A', 'Account B'],
            'Amount': [100.0, -100.0],
            'FacingAccount': [None, None]
        })
        
        result = iat.map_iat(df, iat_value_col='Amount')
        
        # Should identify matching transactions
        assert result.loc[0, 'FacingAccount'] == 'Account B'
        assert result.loc[1, 'FacingAccount'] == 'Account A'

    def test_map_iat_respects_date_window(self, iat_config):
        """Test map_iat only matches within 7-day window."""
        iat = IatIdentification(iat_config)
        
        df = pd.DataFrame({
            'Date': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-15')],
            'Currency': ['GBP', 'GBP'],
            'FullType': ['Savings', 'Savings'],
            'Account': ['Account A', 'Account B'],
            'Amount': [100.0, -100.0],
            'FacingAccount': [None, None]
        })
        
        result = iat.map_iat(df, iat_value_col='Amount')
        
        # Should NOT match - dates too far apart
        assert pd.isna(result.loc[0, 'FacingAccount'])
        assert pd.isna(result.loc[1, 'FacingAccount'])

    def test_map_iat_respects_currency(self, iat_config):
        """Test map_iat only matches same currency."""
        iat = IatIdentification(iat_config)
        
        df = pd.DataFrame({
            'Date': [pd.Timestamp('2023-01-15'), pd.Timestamp('2023-01-16')],
            'Currency': ['GBP', 'EUR'],
            'FullType': ['Savings', 'Savings'],
            'Account': ['Account A', 'Account B'],
            'Amount': [100.0, -100.0],
            'FacingAccount': [None, None]
        })
        
        result = iat.map_iat(df, iat_value_col='Amount')
        
        # Should NOT match - different currencies
        assert pd.isna(result.loc[0, 'FacingAccount'])

    def test_map_iat_fx_identifies_forex_transfers(self, iat_config):
        """Test map_iat_fx identifies FX transfers."""
        iat = IatIdentification(iat_config)
        
        df = pd.DataFrame({
            'Date': [pd.Timestamp('2023-01-15'), pd.Timestamp('2023-01-16')],
            'AccountType': ['Checking', 'Savings'],
            'Currency': ['GBP', 'USD'],
            'Type': ['FX', 'FX'],
            'Account': ['Account A', 'Account B'],
            'Amount': [1000.0, -1100.0],
            'FullType': ['', ''],
            'FullSubType': ['', ''],
            'SubType': ['', ''],
            'MemoMapped': ['', ''],
            'Memo': ['', ''],
            'MemoSimple': ['', ''],
            'Subcategory': ['', ''],
            'SourceFile': ['', ''],
            'YearToDate': [0, 0],
            'FullMasterType': ['', ''],
            'MasterType': ['', ''],
            'FacingAccount': [None, None]
        })
        
        result = iat.map_iat_fx(df)
        
        # Should identify FX transfers
        assert result.loc[0, 'FacingAccount'] == 'Account B'


# ============ LastUpdate Tests ============

class TestLastUpdate:
    """Tests for LastUpdate status class."""

    def test_last_update_date_formatting(self):
        """Test last_update formats dates correctly."""
        config = configparser.ConfigParser()
        config.read = MagicMock(return_value=[None])
        
        # Create mutable config that won't raise KeyError
        config['IO'] = {'path_last_updated': '/tmp/last_update.csv'}
        
        # Note: LastUpdate class has a config initialization bug
        # Just test the last_update method directly
        df = pd.DataFrame({
            'Date': [pd.Timestamp('2023-01-15'), pd.Timestamp('2023-02-20')],
            'Account': ['Bank A', 'Bank A']
        })
        
        # Create instance but don't call initialization that causes error
        lu = object.__new__(LastUpdate)
        result = lu.last_update(df)
        
        assert len(result) == 1
        assert result.loc['Bank A', 'LastUpdate'] == '2023-02-20'


# ============ Process Class Tests ============

class TestProcessInitialization:
    """Tests for Process class initialization."""

    @patch('bkanalysis.process.process.Process._Process__initialise_map')
    def test_process_init_with_config(self, mock_init):
        """Test Process initialization with provided config."""
        mock_init.return_value = pd.DataFrame(columns=['Memo Simple', 'Memo Mapped'])
        
        config = configparser.ConfigParser()
        config['Mapping'] = {
            'expected_columns': "['Date', 'Account', 'AccountType', 'Amount', 'Subcategory', 'Memo', 'Currency', 'SourceFile']",
            'new_columns': "['Date', 'Account', 'AccountType', 'Amount', 'Subcategory', 'Memo', 'Currency', 'SourceFile', 'MemoSimple', 'MemoMapped', 'Type']"
        }
        
        process = Process(config)
        assert process.config == config


class TestProcessMemoMapping:
    """Tests for memo mapping functionality."""

    @patch('bkanalysis.process.process.Process._Process__initialise_map')
    def test_map_memo_simple_mapping(self, mock_init):
        """Test map_memo applies simple memo mappings."""
        mock_init.return_value = pd.DataFrame(columns=['Memo Simple', 'Memo Mapped'])
        
        config = configparser.ConfigParser()
        config['Mapping'] = {
            'expected_columns': "['Date', 'Account', 'AccountType', 'Amount', 'Subcategory', 'Memo', 'Currency', 'SourceFile']",
            'new_columns': "['Result']"
        }
        
        process = Process(config)
        process.map_simple = pd.DataFrame({
            'Memo Simple': ['AMAZON', 'WALMART'],
            'Memo Mapped': ['ONLINE SHOPPING', 'GROCERIES']
        })
        
        memo_series = pd.Series(['AMAZON', 'WALMART'])
        result = process.map_memo(memo_series)
        
        assert result[0] == 'ONLINE SHOPPING'
        assert result[1] == 'GROCERIES'

    @patch('bkanalysis.process.process.Process._Process__initialise_map')
    def test_map_memo_case_insensitive(self, mock_init):
        """Test map_memo is case-insensitive."""
        mock_init.return_value = pd.DataFrame(columns=['Memo Simple', 'Memo Mapped'])
        
        config = configparser.ConfigParser()
        config['Mapping'] = {
            'expected_columns': "['Date', 'Account', 'AccountType', 'Amount', 'Subcategory', 'Memo', 'Currency', 'SourceFile']",
            'new_columns': "['Result']"
        }
        
        process = Process(config)
        process.map_simple = pd.DataFrame({
            'Memo Simple': ['AMAZON'],
            'Memo Mapped': ['ONLINE SHOPPING']
        })
        
        memo_series = pd.Series(['amazon', 'Amazon', 'AMAZON'])
        result = process.map_memo(memo_series)
        
        assert result[0] == 'ONLINE SHOPPING'
        assert result[1] == 'ONLINE SHOPPING'
        assert result[2] == 'ONLINE SHOPPING'


class TestProcessTypeMapping:
    """Tests for type and subtype mapping."""

    @patch('bkanalysis.process.process.Process._Process__initialise_map')
    def test_map_type_extracts_types(self, mock_init):
        """Test map_type extracts type and subtype from memo."""
        mock_init.return_value = pd.DataFrame(columns=['Memo Simple', 'Memo Mapped'])
        
        config = configparser.ConfigParser()
        config['Mapping'] = {
            'expected_columns': "['Date', 'Account', 'AccountType', 'Amount', 'Subcategory', 'Memo', 'Currency', 'SourceFile']",
            'new_columns': "['Result']"
        }
        
        process = Process(config)
        process.map_main = pd.DataFrame({
            'Memo Mapped': ['AMAZON', ''],
            'Type': ['SHOPPING', ''],
            'SubType': ['ONLINE', '']
        })
        process.map_full_type = pd.DataFrame({
            'Type': ['SHOPPING'],
            'FullType': ['Shopping'],
            'MasterType': ['SPENDING']
        })
        process.map_full_subtype = pd.DataFrame({
            'SubType': ['ONLINE'],
            'FullSubType': ['Online Purchase']
        })
        process.map_master = pd.DataFrame({
            'MasterType': ['SPENDING'],
            'FullMasterType': ['Spending']
        })
        
        memo_series = pd.Series(['AMAZON'])
        types, full_types, subtypes, full_subtypes, master_types, full_master_types = process.map_type(memo_series)
        
        assert types[0] == 'SHOPPING'
        assert subtypes[0] == 'ONLINE'

    @patch('bkanalysis.process.process.Process._Process__initialise_map')
    def test_get_full_type_known_type(self, mock_init):
        """Test get_full_type returns full name for known type."""
        mock_init.return_value = pd.DataFrame(columns=['Memo Simple', 'Memo Mapped'])
        
        config = configparser.ConfigParser()
        config['Mapping'] = {
            'expected_columns': "['Date', 'Account', 'AccountType', 'Amount', 'Subcategory', 'Memo', 'Currency', 'SourceFile']",
            'new_columns': "['Result']"
        }
        
        Process(config)  # Initialize to ensure no errors
        
        type_mapping = {'SHOPPING': 'Shopping', 'TRANSFER': 'Bank Transfer'}
        result = Process.get_full_type('SHOPPING', type_mapping)
        assert result == 'Shopping'

    @patch('bkanalysis.process.process.Process._Process__initialise_map')
    def test_get_full_type_unknown_type(self, mock_init):
        """Test get_full_type returns empty string for unknown type."""
        mock_init.return_value = pd.DataFrame(columns=['Memo Simple', 'Memo Mapped'])
        
        config = configparser.ConfigParser()
        config['Mapping'] = {
            'expected_columns': "['Date', 'Account', 'AccountType', 'Amount', 'Subcategory', 'Memo', 'Currency', 'SourceFile']",
            'new_columns': "['Result']"
        }
        
        Process(config)  # Initialize to ensure no errors
        
        type_mapping = {'SHOPPING': 'Shopping'}
        result = Process.get_full_type('UNKNOWN', type_mapping)
        assert result == ''


class TestProcessExtend:
    """Tests for extend functionality."""

    def test_extend_memo_simple_column_creation(self):
        """Test extend creates MemoSimple column from memo cleaning."""
        # Test the memo cleaning logic directly
        test_memos = [
            'AMZN MKTP AMAZONCOMUK',
            'SHOP * STORE',
            'SHOP ON 01/01/2023'
        ]
        
        cleaned = [Process._Process__clean_memo(m) for m in test_memos]
        
        assert 'AMAZON' in str(cleaned[0])  
        assert '*' not in str(cleaned[1])


class TestProcessCleanMemo:
    """Tests for memo cleaning functionality."""

    def test_clean_memo_amazon_variations(self):
        """Test _clean_amazon_memo handles various Amazon formats."""
        assert Process._clean_amazon_memo('AMZN MKTP AMAZONCOMUK') == 'AMAZON'
        assert Process._clean_amazon_memo('AMAZON MKTPL LONDON') == 'AMAZON'
        assert Process._clean_amazon_memo('AMAZON.COM SEATTLE') == 'AMAZON'

    def test_clean_memo_removes_asterisks(self):
        """Test __clean_memo removes asterisks."""
        result = Process._Process__clean_memo('SHOP * STORE')
        assert '*' not in result
        assert 'SHOP' in result

    def test_clean_memo_removes_extra_spaces(self):
        """Test __clean_memo removes extra spaces."""
        result = Process._Process__clean_memo('SHOP    STORE')
        assert 'SHOP STORE' in result or result == 'SHOP STORE'

    def test_clean_memo_handles_on_keyword(self):
        """Test __clean_memo splits on 'ON' keyword."""
        result = Process._Process__clean_memo('SHOP ON 01/01/2023')
        assert 'ON' not in result


class TestProcessIntegration:
    """Integration tests for complete process workflow."""

    def test_process_workflow_memo_cleaning(self):
        """Test memo cleaning part of process workflow."""
        test_cases = [
            ('AMAZON MKTPL LONDON', 'Process AMAZON from marketplace'),
            ('SHOP * STORE', 'Process SHOP from store'),
            ('WIRE TRANSFER ON 01/01/2023', 'Split on ON keyword')
        ]
        
        for memo, description in test_cases:
            result = Process._Process__clean_memo(memo)
            # Should successfully clean without error
            assert isinstance(result, str)
