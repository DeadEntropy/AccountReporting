import pandas as pd
import numpy as np

__EPSILON = 0.01

def clean_nutmeg_activity_report(df_activity):
    df = df_activity[df_activity.Fund.isin(['Djokovic, Nicolas : Rainy day pot', 'Djokovic, Nicolas : My ISA'])]

    df.Value = df.Value.str.replace(',', '').astype('float')
    df.Units = df.Units.str.replace(',', '').astype('float')
    df['Units(S)'] = df['Units(S)'].str.replace(',', '').astype('float')
    df['Value(S)'] = df['Value(S)'].str.replace(',', '').astype('float')

    df['Unitary Value'] = [v/u if u != 0.0 else 0.0 for (u,v) in zip(df['Units(S)'], df['Value(S)'])]

    df = df.set_index('Date', drop=True).sort_index()

    df_piv = pd.DataFrame(pd.pivot_table(df, index=['Date', 'Asset Code', 'Type', 'Narrative'], values=['Units(S)', 'Value(S)', 'Unitary Value'], aggfunc={'Units(S)': sum, 'Value(S)': sum, 'Unitary Value': np.mean}).to_records()).sort_values('Date')
    df_piv = df_piv.rename({'Units(S)':'Units', 'Value(S)':'Value'}, axis=1)
    df_piv = df_piv.set_index('Date', drop=True)

    return df_piv

def get_relevant_purchases_for_sale(sale_units: float, purchases:dict)-> dict: 
    if sale_units == 0:
        return []

    relevant_purchases = {}
    remaining_sale_units = sale_units
    for purchase_date, purchase_unit in purchases.items():
        if remaining_sale_units > purchase_unit:
            relevant_purchases[purchase_date] = purchase_unit
            remaining_sale_units -= purchase_unit
        else:
            relevant_purchases[purchase_date] =  remaining_sale_units
            break

    relevant_sum = sum([v for k, v in relevant_purchases.items()])
    assert abs(sale_units - relevant_sum) < __EPSILON, f"{sale_units} vs {relevant_sum}"
    return relevant_purchases

def get_remaining_purchases(relevant_purchases:dict, purchases:dict)-> dict:
    if len(relevant_purchases) == 0:
        raise Exception(f'relevant_purchases is empty')

    remaining_purchases = {}
    for purchase_date, purchase_unit in purchases.items():
        if purchase_date not in relevant_purchases:
            remaining_purchases[purchase_date] = purchase_unit
        else:
            remaining_purchases[purchase_date] = purchase_unit - relevant_purchases[purchase_date]

    purchase_sum = sum([v for k, v in purchases.items()])
    relevant_remaining_sum = sum([v for k, v in relevant_purchases.items()]) + sum([v for k, v in remaining_purchases.items()])
    assert abs(purchase_sum - relevant_remaining_sum) < __EPSILON, f"{purchase_sum} vs {relevant_remaining_sum}"
    return remaining_purchases

def get_taxable_event_from_single_asset(df: pd.DataFrame) -> pd.DataFrame:
    assert set(df.columns) == set(['Units', 'Unitary Value'])

    transactions_history = {}
    for date in df.index:
        transactions_history[date] = df[df.index <= date].to_dict()

    for date, transactions in transactions_history.items():
        purchases = {}
        sales = {}
        for transaction_date, transaction_unit in transactions['Units'].items():
            if transaction_unit > 0:
                purchases[transaction_date] = transaction_unit
            else:
                sales[transaction_date] = -transaction_unit

        transactions['Purchases'] = purchases
        transactions['Sales'] = sales

        remaining_purchases = purchases
        for sale_unit in list(transactions['Sales'].values())[:-1]:
            remaining_purchases = get_remaining_purchases(get_relevant_purchases_for_sale(sale_unit, remaining_purchases), remaining_purchases)

        if len(transactions['Sales'].values()) > 0:
            relevant_purchase_for_last_sale = get_relevant_purchases_for_sale(list(transactions['Sales'].values())[-1], remaining_purchases)
            remaining_purchases = get_remaining_purchases(relevant_purchase_for_last_sale, remaining_purchases)
            transactions['Relevant_Purchases'] = relevant_purchase_for_last_sale
        else:
            transactions['Relevant_Purchases'] = {}

        transactions['Remaining_Purchases'] = remaining_purchases

    sales = {}

    for transaction_date, transaction in transactions_history.items():
        if transaction_date not in transaction['Sales'].keys():
            continue # this is a purchase, not taxable event

        units_sold = transaction['Sales'][transaction_date]
        sale_price = transaction['Unitary Value'][transaction_date]
        relevant_purchase = transaction['Relevant_Purchases']
        relevant_purchase = [(d, u, transaction['Unitary Value'][d]) for (d, u) in relevant_purchase.items()]

        assert abs(units_sold - sum([u for (d, u, v) in relevant_purchase])) < __EPSILON

        purchase_price = sum([u*v for (d, u, v) in relevant_purchase]) / units_sold
        purchase_date = [d for d, u, v in relevant_purchase if u != 0.0][0]

        sales[transaction_date] = {
            'units_sold': units_sold,
            'sale_price': sale_price,
            'purchase_price': purchase_price,
            'purchase_date': purchase_date,
            }
        
    sales_df = pd.DataFrame.from_dict(sales).transpose()
    sales_df['taxable_amount'] = sales_df['units_sold'] * (sales_df['sale_price'] - sales_df['purchase_price'])
    sales_df['holding_period'] = sales_df.index - sales_df['purchase_date']

    return sales_df
