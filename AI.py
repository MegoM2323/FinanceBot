import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import model_from_json
from keras import layers
import math
from pprint import pprint
import random
import time
import Config
from threading import Thread
import datetime
from binance import ThreadedWebsocketManager
import investpy
import multiprocessing
import pandas_datareader as web
from datetime import datetime
import Informator


'''
    def PredictNextDay(self, LastDays=1, Visualize=False):
        try:
            AI_Models = self.Load_AI()
        except:
            return None
        Base, Predictions, AI_Models = self.Function(AI_Models=AI_Models)
        # print(Predictions[-5:])
        # print(Base[-5:])
        RES = Predictions[-LastDays:][::-1]
        res = []
        for i in range(len(RES)):
            res.append(RES[i][0])
        if Visualize is True:
            plt.style.use('fivethirtyeight')  # специальное отображение графиков для pyplot
            plt.plot(res)
            plt.show()
        return res
        
        

'''


class BlockAI:
    def Create_AI(self, x_train, y_train, epochs=1):
        # Строим нейронку
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        # Компилируем модель
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Тренируем модель
        model.fit(x_train, y_train, batch_size=1, epochs=epochs)

        return model

    def Create_AI_RNN(self, x_train, y_train, epochs=1):
        model = Sequential()
        model.add(layers.Embedding(input_dim=1000, output_dim=64))

        # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
        model.add(layers.GRU(256, return_sequences=True))

        # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
        model.add(layers.SimpleRNN(128))

        model.add(layers.Dense(10))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=1, epochs=epochs)

        return model

    def Save_AI(self, model):
        model_json = model.to_json()
        path = rf"C:\Users\qwert\PycharmProjects\FinanceBot\AI_Models\{self.symbol}_AI.json"
        with open(path, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(rf"C:\Users\qwert\PycharmProjects\FinanceBot\AI_Models\{self.symbol}_model_weights.h5")
        print("Saved model to disk")

    def Load_AI(self):
        try:
            path = rf"C:\Users\qwert\PycharmProjects\FinanceBot\AI_Models\{self.symbol}_AI.json"
        except Exception as E:
            print("Warning:", E)
            return E
        json_file = open(path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(
            rf"C:\Users\qwert\PycharmProjects\FinanceBot\AI_Models\{self.symbol}_model_weights.h5")
        # print("Loaded model from disk")
        return loaded_model

    def Predictions(self, days=1, Visualize=False, AI=None):
        if AI is None:
            model = self.Load_AI()
        else:
            model = AI
        dataset = Informator.get_DataSet(self.symbol).filter(['Close']).values
        # Scale the data (масштабируем)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        Input_Data = np.array([scaled_data[-60:]])
        predictions = scaler.inverse_transform(model.predict(Input_Data))[0][0]
        return predictions

    def Test_Predictions(self, Inputs, Test, AI=None):
        if AI is None:
            model = self.Load_AI()
        else:
            model = AI
        x_train, y_train, x_test, y_test = Inputs
        dataset = y_train
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        Input_Data = np.array([scaler.transform(Test)])
        predictions = scaler.inverse_transform(model.predict(Input_Data))[0][0]
        return predictions

    def prediction_interval(self, prediction, procent):
        res = f'{prediction * (1 - procent / 100)} < {self.symbol} < {prediction * (1 + procent / 100)}'
        print(res)
        return res

    def Get_Input_Data(self, TestData=False):

        Monet = Informator.get_DataSet(self.symbol)

        # Создаем новый датафрейм только с колонкой "Close"
        data = Monet.filter(['Close'])
        # преобразовываем в нумпаевский массив
        dataset = data.values
        # Вытаскиваем количество строк в дате для обучения модели (LSTM)
        training_data_len = math.ceil(len(dataset) * self.split_data_train)

        # Scale the data (масштабируем)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # Создаем датасет для обучения
        train_data = scaled_data[0:training_data_len]
        # разбиваем на x underscore train и y underscore train
        x_train = []
        y_train = []

        # c = 0
        for i in range(60, len(train_data)):
            x_train.append(train_data[i - 60:i])
            y_train.append(train_data[i])
            # if 1600 < c < 1603:
            #    print(11111111, train_data[i - 60:i])
            #    print(22222222, train_data[i])
            # c += 1

        # Конвертируем x_train и y_train в нумпаевский массив
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshape data
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

        if TestData is False:
            return x_train, y_train
        else:
            # Создаем тестовый датасет
            test_data = scaled_data[training_data_len - 60:]
            # по аналогии создаем x_test и y_test
            x_test = []
            y_test = dataset[training_data_len:]
            for i in range(60, len(test_data)):
                x_test.append(test_data[i - 60:i])

            # опять преобразуем в нумпаевский массив
            x_test = np.array(x_test)

            # опять делаем reshape
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

            return x_train, y_train, x_test, y_test

    def Function(self, CoinDataSet, epochs=1, DataPocket=None, AI=None):
        if DataPocket is None:
            x_train, y_train, x_test, y_test = self.Get_Input_Data(TestData=True)
        else:
            x_train, y_train, x_test, y_test = DataPocket

        # Создаем новый датафрейм только с колонкой "Close"
        # преобразовываем в нумпаевский массив
        CoinDataSet = CoinDataSet.filter(['Close']).values
        training_data_len = math.ceil(len(CoinDataSet) * self.split_data_train)

        # Scale the data (масштабируем)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(CoinDataSet)

        if AI is None:
            model = self.Create_AI(x_train, y_train, epochs)
        else:
            model = AI

        # Получаем модель предсказывающую значения
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Получим mean squared error (RMSE) - метод наименьших квадратов
        rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
        print(f'Mean squared error: {rmse}')

        procent_error = round(rmse / Informator.get_price(self.symbol) * 100, 2)
        print(f'Procent error: {procent_error}')

        # Строим график
        plt.style.use('fivethirtyeight')  # специальное отображение графиков для pyplot

        CoinDataSet = list(map(lambda x: round(float(x[0]), 10), CoinDataSet))
        predictions = list(map(lambda x: round(float(x[0]), 10), predictions))
        # Строим график
        # train = CoinDataSet[:training_data_len]
        # print(train)
        # valid = pd.DataFrame()
        # valid['Close'] = CoinDataSet
        # valid['Predictions'] = CoinDataSet[:training_data_len] + predictions

        ## Визуализируем
        # plt.figure(figsize=(16, 8))
        # plt.title('Model LSTM')
        # plt.xlabel('Date', fontsize=18)
        # plt.ylabel('Close Price', fontsize=18)
        ## plt.plot(list(map(lambda x: float(x), train['Close'])))
        # plt.plot(valid[['Close', 'Predictions']])
        # plt.legend(['Train', 'Val', 'Pred'], loc='lower right')
        # plt.show()

        return model, rmse, procent_error

    def Testing_Netwoks(self):
        sl_rmse = {}
        sl_pr = {}
        for symbol in Config.Defaults_symbols:
            try:
                print(symbol)
                BLK = BlockAI(symbol, split_data_train=0.95)
                AI, rmse, pr = BLK.Function(Informator.get_DataSet(symbol), epochs=10, AI=BLK.Load_AI())
                sl_rmse[symbol] = rmse
                sl_pr[symbol] = pr
            except Exception as e:
                sl_rmse[symbol] = 10 ** 100
                sl_pr[symbol] = 10 ** 100
                print(f"Error {symbol}, {e}")
        return sl_rmse, sl_pr

    def Uprage_Networks(self, symbols, sl_rmse, sl_pr):
        while True:
            for symbol in symbols:
                try:
                    print(symbol)
                    BLK = BlockAI(symbol, split_data_train=0.85)
                    AI, rmse, pr = BLK.Function(Informator.get_DataSet(symbol), epochs=10)
                    if rmse < sl_rmse[symbol]:
                        sl_rmse[symbol] = rmse
                        sl_pr[symbol] = pr
                        BLK.Save_AI(AI)
                except Exception as e:
                    print(f"Error {symbol}, {e}")
            pprint(sl_pr)
            pprint(sl_rmse)

    def NowPredictions_ALL(self):
        predictions_price = {}
        intervals_price = {}
        now_price = {}
        for symbol in Config.Defaults_symbols:
            try:
                print(symbol)
                BLK = BlockAI(symbol, split_data_train=0.95)
                AI, rmse, pr = BLK.Function(Informator.get_DataSet(symbol), epochs=10, AI=BLK.Load_AI())
                predictions_price[symbol] = BLK.Predictions()
                intervals_price[symbol] = (predictions_price[symbol] - rmse, predictions_price[symbol] + rmse)
                now_price[symbol] = Informator.get_price(symbol)
            except Exception as e:
                print(f"Error {symbol}, {e}")
        return predictions_price, intervals_price, now_price

    def GET_today_predictions(self):
        with open(r'C:\Users\qwert\PycharmProjects\FinanceBot\docs\QUALITY_PREDICTIONS.txt', 'r') as f:
            lines = f.readlines()
            for i in range(len(lines) - 1, -1, -1):
                if lines[i] == 'FLAG\n':
                    res = lines[i + 1:-1]
                    break
        return ''.join(res)

    def SET_predictions(self, packet):
        predictions_price, intervals_price, now_price = packet
        with open(r'C:\Users\qwert\PycharmProjects\FinanceBot\docs\QUALITY_PREDICTIONS.txt', 'a') as f:
            f.write(f'FLAG\n')
            f.write(f'---------------------------------{datetime.now()}---------------------------------\n')
            for symbol in predictions_price.keys():
                f.write(f'------------{symbol}------------\n')
                f.write(f'Prediction: {predictions_price[symbol]}\n')
                f.write(f'{intervals_price[symbol][0]} < X < {intervals_price[symbol][1]}\n')
                f.write(f'Price was: {now_price[symbol]}\n')
                f.write(f'------------{symbol}------------\n\n')
            f.write(f'---------------------------------{datetime.now()}---------------------------------\n\n\n')

    def DEL_recently_predictions(self):
        with open(r'C:\Users\qwert\PycharmProjects\FinanceBot\docs\QUALITY_PREDICTIONS.txt', 'r') as f:
            lines = f.readlines()
            for i in range(len(lines) - 1, -1, -1):
                if lines[i] == 'FLAG\n':
                    normal_pred = lines[:i]
                    res = lines[i + 1:-1]
                    break
        with open(r'C:\Users\qwert\PycharmProjects\FinanceBot\docs\QUALITY_PREDICTIONS.txt', 'w') as f:
            f.writelines(normal_pred)

        return ''.join(res)

    def __init__(self, symbol, split_data_train=0.8):
        self.symbol = symbol
        # Until 1
        self.split_data_train = split_data_train

class FB_AI(BlockAI):
    pass


'''symbol = "ETC"
BLK = BlockAI(symbol, split_data_train=0.8)
#AI = BLK.Load_AI()
AI, rmse, pr = BLK.Function(Informator.get_DataSet(symbol), epochs=10)
BLK.Save_AI(AI)
pred = BLK.Predictions()
BLK.prediction_interval(pred, rmse)'''
