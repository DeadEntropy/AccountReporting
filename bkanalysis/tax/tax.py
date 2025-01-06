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
    "2020": {"Basic": TaxBand(0, 37500, 0.2), "Higher": TaxBand(37500, 150000, 0.4), "Additional": TaxBand(150000, None, 0.45)},
    "2021": {"Basic": TaxBand(0, 37700, 0.2), "Higher": TaxBand(37700, 150000, 0.4), "Additional": TaxBand(150000, None, 0.45)},
}


def compute_uk_tax_return(total_taxable, tax_already_paid, tax_year="2021", tax_bands=None):
    if tax_bands is None:
        tax_bands = __tax_bands[tax_year]

    results = {}
    tax_due = 0.0
    for key, tax_band in tax_bands.items():
        tax_due_in_band = tax_band.tax_to_pay(total_taxable)
        results[key] = [tax_band, tax_due_in_band]
        tax_due += tax_due_in_band

    return tax_due - tax_already_paid, results


class PayrollData:

    def __init__(self, json):
        self._json = json

    @property
    def earnings(self):
        return sum([v for v in self._json["Earnings"].values()])

    @property
    def taxable_benefits(self):
        return sum([v for v in self._json["Taxable Benefits"].values()])

    @property
    def pre_tax_deductions(self):
        return sum([v for v in self._json["Pre-Tax Deductions"].values()])

    @property
    def taxes(self):
        return sum([v for v in self._json["Taxes"].values()])

    @property
    def post_tax_deductions(self):
        return sum([v for v in self._json["Post-Tax Deductions"].values()])

    @property
    def reimbursements(self):
        return sum([v for v in self._json["Reimbursements"].values()])

    @property
    def net_pay(self):
        return self.earnings + self.taxable_benefits - self.pre_tax_deductions - self.taxes - self.post_tax_deductions + self.reimbursements

    def waterfall_chart(self):
        import plotly.graph_objects as go

        _y = [
            self.earnings,
            self.taxable_benefits,
            -self.pre_tax_deductions,
            -self.taxes,
            -self.post_tax_deductions,
            self.reimbursements,
            self.net_pay,
        ]
        fig = go.Figure(
            go.Waterfall(
                name="20",
                orientation="v",
                measure=["relative", "relative", "relative", "relative", "relative", "relative", "total"],
                x=["Earnings", "Taxable Benefits", "Pre-Tax Deductions", "Taxes", "Post-Tax Deductions", "Reimbursements", "Net Pay"],
                textposition="outside",
                text=[f"{v:,.0f}" for v in _y],
                y=_y,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            )
        )

        fig.update_layout(title="Payroll Breakdown", showlegend=True)

        return fig


class Payrolls:
    def __init__(self, payrolls: list):
        self._payrolls = payrolls

    @property
    def earnings(self):
        return sum([p.earnings for p in self._payrolls])

    @property
    def taxable_benefits(self):
        return sum([p.taxable_benefits for p in self._payrolls])

    @property
    def pre_tax_deductions(self):
        return sum([p.pre_tax_deductions for p in self._payrolls])

    @property
    def taxes(self):
        return sum([p.taxes for p in self._payrolls])

    @property
    def post_tax_deductions(self):
        return sum([p.post_tax_deductions for p in self._payrolls])

    @property
    def reimbursements(self):
        return sum([p.reimbursements for p in self._payrolls])

    @property
    def net_pay(self):
        return sum([p.net_pay for p in self._payrolls])


def waterfall_chart(payroll, inflows):
    import plotly.graph_objects as go

    _x = ["Earnings", "Taxable Benefits", "Pre-Tax Deductions", "Taxes", "Post-Tax Deductions", "Reimbursements", "Net Pay"]
    _y = [
        payroll.earnings,
        payroll.taxable_benefits,
        -payroll.pre_tax_deductions,
        -payroll.taxes,
        -payroll.post_tax_deductions,
        payroll.reimbursements,
        payroll.net_pay,
    ]
    _measure = ["relative", "relative", "relative", "relative", "relative", "relative", "total"]
    _text = [f"{v:,.0f}" for v in _y]

    if inflows is not None:
        net_pay_discrepancy = inflows["Salary"] - payroll.net_pay

        _x = _x + ["Discrepancy"] + list(inflows.index)
        _y = _y + [net_pay_discrepancy] + list(inflows.values)
        _measure = _measure + ["relative", "total"] + ["relative" for _ in inflows[1:].index]
        _text = _text + [f"{net_pay_discrepancy:,.0f}"] + [f"{v:,.0f}" for v in inflows.values]

    fig = go.Figure(
        go.Waterfall(
            name="",
            orientation="v",
            measure=_measure,
            x=_x,
            textposition="outside",
            text=_text,
            y=_y,
            hoverinfo=None,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )

    fig.update_layout(
        title="Payroll Breakdown",
    )

    return fig
