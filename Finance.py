from pprint import pprint

import pandas as pd

from AI import BlockAI
import Config
from threading import Thread
import Informator
from binance import ThreadedWebsocketManager
import binance
from datetime import datetime
import time
from matplotlib import pyplot as plt

client = 0  # binance.Client(Config.ShifrKey, Config.ShifrSecretKey)
# twm = ThreadedWebsocketManager(api_key=Config.ShifrKey, api_secret=Config.ShifrSecretKey)
# twm.start()

"""
 Сделать фьючерсный кошелек
"""




class Wallet:
    def __init__(self, commission=Config.commission):
        self.commission = commission
        self.coins = {'USDT': 0}

    def buy_coin_usd(self, symbol, usd):
        if self.coins['USDT'] >= usd >= 11:
            price = Informator.get_price(symbol)
            if symbol not in self.coins.keys():
                self.coins[symbol] = usd / price
                self.coins['USDT'] -= usd + usd * self.commission
            else:
                self.coins[symbol] += usd / price
                self.coins['USDT'] -= usd + usd * self.commission
            if Config.TestMode is False:
                OrderBay = client.order_market_buy(
                    symbol=symbol + 'USDT',
                    quantity=usd / price
                )
        else:
            if usd < 11:
                print(f'Слишком маленькая сумма {symbol}')
            else:
                print(f'Недостаточно средств {symbol}')

    def sell_coin_usd(self, symbol, usd):
        price = Informator.get_price(symbol)
        if symbol not in self.coins.keys():
            print('В кошелке такой монеты нет')
        elif self.coins[symbol] >= usd / price > 0 and usd >= 11:
            self.coins[symbol] -= usd / price
            self.coins['USDT'] += usd - usd * self.commission
            if Config.TestMode is False:
                OrderBay = client.order_market_sell(
                    symbol=symbol + 'USDT',
                    quantity=usd / price
                )
        else:
            print(f'Недостаточно средств {symbol}')

    def buy_coin_quantity(self, symbol, quantity):
        price = Informator.get_price(symbol)
        if self.coins['USDT'] >= quantity * price >= 11:
            if symbol not in self.coins.keys():
                self.coins[symbol] = quantity
                self.coins['USDT'] -= quantity * price + quantity * price * self.commission
            else:
                self.coins[symbol] += quantity
                self.coins['USDT'] -= quantity * price + quantity * price * self.commission
            if Config.TestMode is False:
                OrderBuy = client.order_market_buy(
                    symbol=symbol + 'USDT',
                    quantity=quantity
                )
        else:
            if quantity * price < 11:
                print(f'Слишком маленькая сумма {symbol}')
            else:
                print(f'Недостаточно средств {symbol}')

    def sell_coin_quantity(self, symbol, quantity):
        price = Informator.get_price(symbol)
        if symbol not in self.coins.keys():
            print('В кошелке такой монеты нет')
        elif self.coins[symbol] >= quantity > 0 and quantity * price >= 11:
            self.coins[symbol] -= quantity
            self.coins['USDT'] += quantity * price - quantity * price * self.commission
            if Config.TestMode is False:
                OrderSell = client.order_market_sell(
                    symbol=symbol + 'USDT',
                    quantity=quantity
                )
        else:
            print(f'Недостаточно средств {symbol}')

    def deposit(self, quantity, symbol='USDT'):
        if symbol not in self.coins.keys():
            self.coins[symbol] = quantity
        else:
            self.coins[symbol] += quantity

    def withdraw(self, quantity, symbol='USDT'):
        if symbol not in self.coins.keys():
            print('В кошелке такой монеты нет')
        else:
            self.coins[symbol] -= quantity

    def ALL_IN_USD(self):
        all_usd = self.coins["USDT"]
        for coin in self.coins:
            if coin != 'USDT':
                all_usd += self.coins[coin] * Informator.get_price(coin)
        return all_usd

    def __str__(self):
        r1 = f'USDT: {self.coins["USDT"]}'
        r2 = f'Portfolio: {self.coins}'
        all_usd = self.coins["USDT"]
        for coin in self.coins:
            if coin != 'USDT':
                all_usd += self.coins[coin] * Informator.get_price(coin)
        r3 = f'USD_ALL: {all_usd}'
        return r1 + '\n' + r2 + '\n' + r3

    def __getitem__(self, symbol):
        return self.coins[symbol]


class FutureWallet(Wallet):
    def __init__(self, shoulder, commission=Config.commission):
        self.commission = commission
        self.shoulder = shoulder
        self.coins = {'USDT': 0}

    def buy_coin_usd(self, symbol, usd):
        if self.coins['USDT'] * self.shoulder >= usd >= 11:
            price = Informator.get_price(symbol)
            if symbol not in self.coins.keys():
                self.coins[symbol] = usd / price
                self.coins['USDT'] -= usd + usd * self.commission
            else:
                self.coins[symbol] += usd / price
                self.coins['USDT'] -= usd + usd * self.commission
            if Config.TestMode is False:
                OrderBay = client.order_market_buy(
                    symbol=symbol + 'USDT',
                    quantity=usd / price
                )
        else:
            if usd < 11:
                print(f'Слишком маленькая сумма {symbol}')
            else:
                print(f'Недостаточно средств {symbol}')

    def sell_coin_usd(self, symbol, usd):
        price = Informator.get_price(symbol)
        if symbol not in self.coins.keys():
            print('В кошелке такой монеты нет')
        elif self.coins[symbol] >= usd / price > 0 and usd >= 11:
            self.coins[symbol] -= usd / price
            self.coins['USDT'] += usd - usd * self.commission
            if Config.TestMode is False:
                OrderBay = client.order_market_sell(
                    symbol=symbol + 'USDT',
                    quantity=usd / price
                )
        else:
            print(f'Недостаточно средств {symbol}')

    def sell_coin_quantity(self, symbol, quantity):
        price = Informator.get_price(symbol)
        if symbol not in self.coins.keys():
            print('В кошелке такой монеты нет')
        elif self.coins[symbol] >= quantity > 0 and quantity * price >= 11:
            self.coins[symbol] -= quantity
            self.coins['USDT'] += quantity * price - quantity * price * self.commission
            if Config.TestMode is False:
                OrderSell = client.order_market_sell(
                    symbol=symbol + 'USDT',
                    quantity=quantity
                )
        else:
            print(f'Недостаточно средств {symbol}')

    def Check_Liquadation(self, prices={}):
        all_coins_usd = 0
        for coin in self.coins:
            if coin != 'USDT':
                if prices == {}:
                    all_coins_usd += self.coins[coin] * Informator.get_price(coin)
                else:
                    all_coins_usd += self.coins[coin] * prices[coin]
        if all_coins_usd + self['USDT'] > 0:
            return False
        return True



class TestModeWallet(Wallet):
    def buy_coin_usd(self, symbol, usd, price):
        if self.coins['USDT'] >= usd >= 11:
            if symbol not in self.coins.keys():
                self.coins[symbol] = usd / price
                self.coins['USDT'] -= usd + usd * self.commission
            else:
                self.coins[symbol] += usd / price
                self.coins['USDT'] -= usd + usd * self.commission
        else:
            if usd < 11:
                print(f'Слишком маленькая сумма {symbol}')
            else:
                print(f'Недостаточно средств {symbol}')

    def sell_coin_usd(self, symbol, usd, price):
        if symbol not in self.coins.keys():
            print('В кошелке такой монеты нет')
        elif self.coins[symbol] >= usd / price > 0 and usd >= 11:
            self.coins[symbol] -= usd / price
            self.coins['USDT'] += usd - usd * self.commission
        else:
            print(f'Недостаточно средств {symbol}')

    def buy_coin_quantity(self, symbol, quantity, price):
        if self.coins['USDT'] >= quantity * price >= 11:
            if symbol not in self.coins.keys():
                self.coins[symbol] = quantity
                self.coins['USDT'] -= quantity * price + quantity * price * self.commission
            else:
                self.coins[symbol] += quantity
                self.coins['USDT'] -= quantity * price + quantity * price * self.commission
        else:
            if quantity * price < 11:
                print(f'Слишком маленькая сумма {symbol}')
            else:
                print(f'Недостаточно средств {symbol}')

    def sell_coin_quantity(self, symbol, quantity, price):
        if symbol not in self.coins.keys():
            print('В кошелке такой монеты нет')
        elif self.coins[symbol] >= quantity > 0 and quantity * price >= 11:
            self.coins[symbol] -= quantity
            self.coins['USDT'] += quantity * price - quantity * price * self.commission
        else:
            print(f'Недостаточно средств {symbol}')

    def ALL_IN_USD(self, prices):
        all_usd = self.coins["USDT"]
        for coin in self.coins:
            if coin != 'USDT':
                if coin in prices.keys():
                    all_usd += self.coins[coin] * prices[coin]
                else:
                    print(f'Монета {coin} не учтена')
        return all_usd


class Instruments:
    def create_SMA(self, data, days):
        SMA = [0] * (days + 1)
        value = sum(data[:days]) / days
        for i in range(days, len(data) - 1):
            SMA.append(value)
            value -= data[i - days] / days
            value += data[i + 1] / days
        return SMA

    def create_EMA(self, data, days):
        EMA = []
        EMA.append(data[0])
        coef = 2 / (days + 1)
        for i in range(len(data) - 1):
            value = coef * data[i] + (1 - coef) * EMA[i - 1]
            EMA.append(value)
        return EMA

    def Test_MA_strategy(self, symbol, data, MA_list, function, Freeze_days=1, reverse=False, Visualise=False):
        df = data.copy()
        for i in MA_list:
            df[f'MA_{i}'] = function(df['Close'], i)
        df['Cross'] = 0
        flag = reverse

        def CrossCheck(S1, S2, i):
            if df[f'MA_{S1}'][i - 1] > df[f'MA_{S2}'][i - 1] and df[f'MA_{S1}'][i] < df[f'MA_{S2}'][i]:
                return True
            elif df[f'MA_{S1}'][i - 1] < df[f'MA_{S2}'][i - 1] and df[f'MA_{S1}'][i] > df[f'MA_{S2}'][i]:
                return True

        for i in range(1, df.shape[0]):
            for j in range(1, len(MA_list)):
                if CrossCheck(MA_list[j - 1], MA_list[j], i):
                    df['Cross'].iat[i] = 1

        if Visualise:
            figure = plt.figure()
            figure.clear()
            ax1 = plt.subplot(111)
            ax1.grid(True)

        w = TestModeWallet()
        w.deposit(1000)
        for i in range(Freeze_days, df.shape[0]):
            try:
                if df['Cross'].iloc[i] == 1:
                    if flag:
                        flag = False
                        plt.axvline(x=df.index[i], color='r')
                        w.sell_coin_quantity(
                            symbol=symbol,
                            quantity=w[symbol],
                            price=df['Close'].iloc[i]
                        )
                    elif flag is False:
                        flag = True
                        plt.axvline(x=df.index[i], color='g')
                        w.buy_coin_usd(
                            symbol=symbol,
                            usd=w['USDT'],
                            price=df['Close'].iloc[i]
                        )
            except:
                pass

        if Visualise:
            ax1.set_yscale('log')
            lines = ['Close'] + [f'MA_{i}' for i in MA_list]
            ax1.plot(df[lines])

        res = w.ALL_IN_USD({symbol: df['Close'].iloc[-1]})
        procent = round((res - 1000) / 10, 2)
        return procent

    def MA_end_check(self, symbol, MA_list, check_days=1, function=create_EMA, name=''):
        df = pd.DataFrame()
        df['Close'] = pd.read_csv(f'Minutes_DataSets\\{symbol + name}')['Close']
        for i in MA_list:
            df[f'MA_{i}'] = function(self, df['Close'], i)
        df['Cross'] = 0

        def CrossCheck(S1, S2, i):
            if df[f'MA_{S1}'][i - 1] > df[f'MA_{S2}'][i - 1] and df[f'MA_{S1}'][i] < df[f'MA_{S2}'][i]:
                return True
            elif df[f'MA_{S1}'][i - 1] < df[f'MA_{S2}'][i - 1] and df[f'MA_{S1}'][i] > df[f'MA_{S2}'][i]:
                return True

        for i in range(1, df.shape[0]):
            for j in range(1, len(MA_list)):
                if CrossCheck(MA_list[j - 1], MA_list[j], i):
                    df['Cross'].iat[i] = 1

        if sum(df['Cross'].iloc[-check_days:]) > 0:
            return True
        else:
            return False

    def MA_optimizator(self, data):
        sl = []
        for i in range(5, 120, 3):
            for j in range(i + 1, 120, 3):
                for k in range(j + 2, 120, 3):
                    MA_list = [i, j, k]
                    res = self.Test_MA_strategy(
                        symbol='DEFUALT',
                        data=data,
                        MA_list=MA_list,
                        reverse=True,
                        function=self.create_EMA,
                    )
                    sl.append([MA_list, res])
                    print(sl[-1])
        return sl[-10:]


class BrainTrader:
    def __init__(self, coins, split_data_dataset=0.90):
        self.symbols = coins
        self.DATASETS = {}
        self.BLKS = {}
        for coin in coins:
            self.BLKS[coin] = BlockAI(coin, split_data_train=split_data_dataset)
            self.DATASETS[coin] = self.BLKS[coin].Get_Input_Data(TestData=True)

    def Test_Trade(self, symbol, coeff=1.05, wallet=None):
        x_train, y_train, x_test, y_test = self.DATASETS[symbol]
        if wallet is None:
            wallet = TestModeWallet()
            wallet.deposit(1000, symbol)
            wallet.deposit(1000 * y_test[0][0])
        start_USD = wallet.ALL_IN_USD({symbol: y_test[0][0]})
        end_USD = wallet.ALL_IN_USD({symbol: y_test[-1][0]})
        step_usd = wallet.ALL_IN_USD({symbol: y_test[0][0]}) / 3
        print(start_USD, end_USD)
        print(((end_USD - start_USD) / start_USD) * 100)
        DELTA_PROCENT_NOT_DO_ANYTHING = round(((end_USD - start_USD) / start_USD) * 100, 2)

        AI, rmse, pr = self.BLKS[symbol].Function(Informator.get_DataSet(symbol), epochs=10,
                                                  AI=self.BLKS[symbol].Load_AI())
        list_profit_loss_ratio = []
        list_all_in_usd = []

        def action(prediction, now_price):
            profit_ratio = abs(prediction + rmse - now_price) / abs(prediction - rmse - now_price)[0]
            loss_ratio = abs(prediction - rmse - now_price) / abs(prediction + rmse - now_price)[0]
            if profit_ratio > coeff:
                print('BUY')
                wallet.buy_coin_usd(symbol, step_usd, price=now_price)
            elif loss_ratio > coeff:
                print('SELL')
                wallet.sell_coin_usd(symbol, step_usd, price=now_price)
            else:
                print('DO NOTHING')
            all_in_usd = wallet.ALL_IN_USD(prices={symbol: now_price})

            print("Profit_ratio:", profit_ratio)
            print("Loss_ratio:", loss_ratio)
            list_all_in_usd.append(round(float(all_in_usd), 2))
            list_profit_loss_ratio.append((float(profit_ratio[0]), float(loss_ratio[0])))

        for i in range(len(x_test)):
            prediction = self.BLKS[symbol].Test_Predictions(Inputs=[x_train, y_train, x_test, y_test], Test=x_test[i],
                                                            AI=AI)
            print("Predictions", prediction)
            price = y_test[i]
            print("Price", price)
            action(prediction, price)

        end_USD = wallet.ALL_IN_USD({symbol: y_test[-1][0]})[0]
        DELTA_PROCENT_NOT_DO_SOMETHING = round(((end_USD - start_USD) / start_USD) * 100, 2)

        return list_all_in_usd, list_profit_loss_ratio, DELTA_PROCENT_NOT_DO_ANYTHING, DELTA_PROCENT_NOT_DO_SOMETHING

    def Trade(self, wallet, symbol, step_usd=15, coeff=15):
        print('Start Trade')
        print(wallet)
        # ____________Prepare________________---
        AI, rmse, pr = self.BLKS[symbol].Function(Informator.get_DataSet(symbol), epochs=10,
                                                  AI=self.BLKS[symbol].Load_AI())
        dates_list = [
            datetime(2022, 10, 10, 1, 0),
            datetime(2022, 10, 10, 6, 0),
            datetime(2022, 10, 10, 11, 0),
            datetime(2022, 10, 10, 16, 0),
            datetime(2022, 10, 10, 21, 0),
        ]
        now = datetime.now()

        # ____________Prepare________________---
        def action():
            prediction = self.BLKS[symbol].Predictions()
            now_price = Informator.get_price(symbol)
            profit_ratio = abs(prediction + rmse - now_price) / abs(prediction - rmse - now_price)
            loss_ratio = abs(prediction - rmse - now_price) / abs(prediction + rmse - now_price)
            if profit_ratio > coeff:
                wallet.buy_coin_usd(symbol, step_usd)
            elif loss_ratio > coeff:
                wallet.sell_coin_usd(symbol, step_usd)
            return prediction, profit_loss_ratio

        prediction, profit_loss_ratio = action()
        while True:
            for date in dates_list:
                if date.hour == now.hour and 25 < now.minute < 45:
                    prediction, profit_loss_ratio = action()

            print(datetime.now().strftime('%D-%H-%M'))
            print(f'Predictions: {prediction}')
            print(f'profit_loss_ratio: {profit_loss_ratio}')
            print(wallet)
            with open('docs/diary.txt', 'a') as f:
                f.write(datetime.now().strftime('%D-%H-%M'))
                f.write(f'Predictions: {prediction}')
                f.write(f'profit_loss_ratio: {profit_loss_ratio}')
                f.write(str(wallet))
                f.write('\n')
            time.sleep(60 * 60)

    def Check(self, coeffs, path='docs\strategy_tests_result.txt'):
        for symbol in self.symbols:
            with open(path, 'a') as f:
                f.write(f'------------------------------{symbol}------------------------------\n')
                USD, PLR, ND, DS = self.Test_Trade(symbol=symbol)
                f.write(f'DAYS: {len(USD)}\n')
                f.write(f'Do nothing {ND}\n')
            for coeff in coeffs:
                try:
                    USD, PLR, ND, DS = self.Test_Trade(symbol=symbol, coeff=coeff)
                    with open(path, 'a') as f:
                        f.write(f'-----------{coeff}-----------\n')
                        f.write(f'Do something {DS}\n')
                    print('USD')
                    pprint(USD)
                    print('PLR')
                    pprint(PLR)
                    print('DAYS', len(USD))
                    print('Delta_Procents', ND, DS)
                except:
                    with open(path, 'a') as f:
                        f.write(f'-----------{coeff}-----------\n')
                        f.write(f'------------ERROR-------------\n')

            with open(path, 'a') as f:
                f.write(f'------------------------------{symbol}------------------------------\n\n\n')
