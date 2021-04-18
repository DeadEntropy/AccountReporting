import unittest
from bkanalysis.tax import tax


class TestTax(unittest.TestCase):
    def test_tax_basic_band(self):
        basic_band = tax.TaxBand(0, 37500, 0.2)
        self.assertEqual(basic_band.tax_to_pay(200000), 7500)

    def test_tax_higher_band(self):
        higher_band = tax.TaxBand(37500, 150000, 0.4)
        self.assertEqual(higher_band.tax_to_pay(200000), 45000)

    def test_tax_additional_band(self):
        advanced_band = tax.TaxBand(150000, None, 0.45)
        self.assertEqual(advanced_band.tax_to_pay(200000), 22500)

    def test_compute_uk_tax_return(self):
        results, details = tax.compute_uk_tax_return(200000, 0.0, tax_year='2020', tax_bands=None)
        self.assertEqual(results, 75000)

    def test_compute_uk_tax_return_2(self):
        results, details = tax.compute_uk_tax_return(200000, 70000, tax_year='2020', tax_bands=None)
        self.assertEqual(results, 5000)

    def test_compute_uk_tax_return_3(self):
        results, details = tax.compute_uk_tax_return(200000, 80000, tax_year='2020', tax_bands=None)
        self.assertEqual(results, -5000)


if __name__ == '__main__':
    unittest.main()
