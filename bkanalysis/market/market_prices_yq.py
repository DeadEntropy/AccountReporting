from yahooquery import Ticker

def get_currency(ticker_list: list):
    all_symbols = " ".join(ticker_list)
    myInfo = Ticker(all_symbols)
    return {p: myInfo.price[p]['currency'] for p in myInfo.price}