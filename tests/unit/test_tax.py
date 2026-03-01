"""Tests for UK tax calculations."""
import pytest

from bkanalysis.tax import tax


class TestTax:
    """Tests for tax band and UK tax return calculations."""

    def test_tax_basic_band(self):
        """Verify basic tax band calculation."""
        basic_band = tax.TaxBand(0, 37500, 0.2)
        assert basic_band.tax_to_pay(200000) == 7500

    def test_tax_higher_band(self):
        """Verify higher tax band calculation."""
        higher_band = tax.TaxBand(37500, 150000, 0.4)
        assert higher_band.tax_to_pay(200000) == 45000

    def test_tax_additional_band(self):
        """Verify additional tax band calculation."""
        advanced_band = tax.TaxBand(150000, None, 0.45)
        assert advanced_band.tax_to_pay(200000) == 22500

    def test_compute_uk_tax_return(self):
        """Verify UK tax return calculation for 2020 tax year (no allowance)."""
        results, details = tax.compute_uk_tax_return(200000, 0.0, tax_year="2020", tax_bands=None)
        assert results == 75000

    def test_compute_uk_tax_return_2(self):
        """Verify UK tax return calculation with £70k allowance."""
        results, details = tax.compute_uk_tax_return(200000, 70000, tax_year="2020", tax_bands=None)
        assert results == 5000

    def test_compute_uk_tax_return_3(self):
        """Verify UK tax return calculation with £80k allowance."""
        results, details = tax.compute_uk_tax_return(200000, 80000, tax_year="2020", tax_bands=None)
        assert results == -5000
