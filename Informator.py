import datetime
import investpy
import numpy as np
import pandas as pd
import time
from threading import Thread
from pycoingecko import CoinGeckoAPI

import Config
import pandas_datareader as web
from binance.client import Client


def get_DataSet(symbol):
    start = datetime.datetime(2010, 5, 1)
    end = datetime.datetime.now()
    try:
        data = web.DataReader(f"{symbol}-USD", "yahoo", start, end)
    except Exception as E1:
        print('Неполучена цена:', E1)
        try:
            Investpy_Data_start = str(start).split()[0].split('-')[::-1]
            Investpy_Data_end = str(end).split()[0].split('-')[::-1]
            data = investpy.get_crypto_historical_data(crypto=symbol,
                                                       from_date='/'.join(Investpy_Data_start),
                                                       to_date='/'.join(Investpy_Data_end))
        except Exception as E2:
            print('Неполучена цена:', E2)
    return data


def get_price(symbol):
    cg = CoinGeckoAPI()
    price = cg.get_price(symbol, vs_currencies='usd')[symbol]['usd']
    #client = Client(Config.ShifrKey, Config.ShifrSecretKey)
    #price = client.get_avg_price(symbol=f'{symbol}USDT')['price']
    return float(price)


def create_minute_DataSet(symbol, offset_time=60*5 - 2, name='', start_with_zero=True):
    def main_function():
        df = pd.DataFrame()
        if start_with_zero is False:
            df['Close'] = pd.read_csv(f'Minutes_DataSets\\{symbol}')['Close']
        else:
            df['Close'] = np.array([])
        df.loc[len(df)] = [get_price(symbol)]
        while True:
            now_price = get_price(symbol)
            time.sleep(offset_time)
            print(now_price)
            df.loc[len(df)] = [now_price]
            df.to_csv(f'Minutes_DataSets/{symbol}{name}')

    t = Thread(target=main_function)
    t.start()

#print(get_price('bitcoin'))

