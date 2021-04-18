
class TaxBand:
    def __init__(self, low, high, rate):
        self.Low = low
        self.High = high
        self.Rate = rate

    def tax_to_pay(self, amount):
        if self.High is None:
            return max(0, amount - self.Low) * self.Rate

        amount_in_band = min(self.High - self.Low, max(0, amount - self.Low))
        return amount_in_band * self.Rate


__tax_bands = {
    '2020': {
        'Basic': TaxBand(0, 37500, 0.2),
        'Higher': TaxBand(37500, 150000, 0.4),
        'Additional': TaxBand(150000, None, 0.45)
    },
    '2021': {
        'Basic': TaxBand(0, 37700, 0.2),
        'Higher': TaxBand(37700, 150000, 0.4),
        'Additional': TaxBand(150000, None, 0.45)
    },
}


def compute_uk_tax_return(total_taxable, tax_already_paid, tax_year='2021', tax_bands=None):
    if tax_bands is None:
        tax_bands = __tax_bands[tax_year]

    results = {}
    tax_due = 0.0
    for key, tax_band in tax_bands.items():
        tax_due_in_band = tax_band.tax_to_pay(total_taxable)
        results[key] = [tax_band, tax_due_in_band]
        tax_due += tax_due_in_band

    return tax_due - tax_already_paid, results
