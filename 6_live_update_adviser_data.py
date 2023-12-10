"""
    В этом коде мы формируем файл с предсказаниями, который требуется для торгового советника,

    Авторы: Олег Шпагин, Федор Шпагин
    Github: https://github.com/WISEPLAT
    Telegram: https://t.me/OlegSh777
"""

# git clone https://github.com/WISEPLAT/backtrader_moexalgo
# pip install backtrader

import functions
import numpy as np
import pandas as pd
import datetime as dt
import backtrader as bt
from backtrader_moexalgo.backtrader_moexalgo.moexalgo_store import MoexAlgoStore  # Хранилище AlgoPack
from Strategy import StrategyJustPrintsOHLCVAndSuperCandles  # Торговая система
from my_config.trade_config import Config  # Файл конфигурации

# Несколько тикеров для нескольких торговых систем по одному временнОму интервалу history + live
if __name__ == '__main__':  # Точка входа при запуске этого скрипта

    symbols = Config.training_NN  # тикеры, по которым будем получать данные для обновления рекомендаций

    store = MoexAlgoStore()  # Хранилище AlgoPack
    cerebro = bt.Cerebro(quicknotify=True)

    for symbol in symbols:  # Пробегаемся по всем тикерам

        # 1. Исторические 5-минутные бары за последние 100 часов + График т.к. оффлайн/ таймфрейм M5
        fromdate = dt.datetime.utcnow() - dt.timedelta(minutes=100*60)  # берем данные за последние 100 часов
        # 9. Исторические 5 минутные бары + Super Candles (tradestats: history M5)
        data = store.getdata(dataname=symbol, timeframe=bt.TimeFrame.Minutes, compression=5, fromdate=fromdate, live_bars=False,
                             super_candles=True, # для получения свечей SuperCandles с расширенным набором характеристик
                             metric='tradestats',  # + необходимо указать тип получаемых метрик
                             )
        # # 11. Исторические 5 минутные бары + Super Candles (orderstats: history + live M5)  // Без данных OHLCV == 0.0, т.к. эти данные можно получить 2-м потоком
        data2 = store.getdata(dataname=symbol, timeframe=bt.TimeFrame.Minutes, compression=5, fromdate=fromdate, live_bars=False,
                             super_candles=True,  # для получения свечей SuperCandles с расширенным набором характеристик
                             metric='orderstats',  # + необходимо указать тип получаемых метрик
                             )

        # # 12. Исторические 5 минутные бары + Super Candles (obstats: history + live M5)  // Без данных OHLCV == 0.0, т.к. эти данные можно получить 2-м потоком
        data3 = store.getdata(dataname=symbol, timeframe=bt.TimeFrame.Minutes, compression=5, fromdate=fromdate, live_bars=False,
                             super_candles=True,  # для получения свечей SuperCandles с расширенным набором характеристик
                             metric='obstats',  # + необходимо указать тип получаемых метрик
                             )

        cerebro.adddata(data)  # Добавляем данные
        cerebro.adddata(data2)  # Добавляем данные
        cerebro.adddata(data3)  # Добавляем данные

    cerebro.addstrategy(StrategyJustPrintsOHLCVAndSuperCandles)  # Добавляем торговую систему

    results = cerebro.run()  # Запуск торговой системы

    # symbol = 'SBER'
    _data = {}
    _df_all = {}
    _percents = {}

    for symbol in symbols:

        df_1 = pd.DataFrame(results[0].supercandles[symbol]['tradestats'])
        _percents[symbol] = functions.get_percent(df_1.copy())
        # print("_percents[symbol]", _percents[symbol])
        _ = df_1.pop("datetime")
        # df_1.set_index(['datetime'], inplace=True)
        # print(df_1)

        df_2 = pd.DataFrame(results[0].supercandles[symbol]['orderstats'])
        _ = df_2.pop("datetime")
        # df_2.set_index(['datetime'], inplace=True)
        # print(df_2)

        df_3 = pd.DataFrame(results[0].supercandles[symbol]['obstats'])
        _ = df_3.pop("datetime")
        # df_3.set_index(['datetime'], inplace=True)
        # print(df_3)

        df_123 = pd.concat([df_1, df_2, df_3], axis=1)
        # print(df_123)

        df_4 = pd.DataFrame(results[0].candles[symbol], columns=["datetime", "open", "high", "low", "close", "volume"])
        # print(df_4)

        df_1234 = pd.concat([df_123, df_4], axis=1)
        df_1234["d_close"] = df_1234["close"].diff()  # чтобы смотреть закрытие выше предыдущего или ниже
        df_1234["d_close"] = np.where(df_1234['d_close'] > 0, 1, 0)
        df_1234.dropna(inplace=True)
        # print(df_1234)

        _data_nn = [df_1234["pr_change"].iloc[-1], df_1234["put_vol_b"].iloc[-1], df_1234["imbalance_vol"].iloc[-1]]
        # print(_data_nn)

        _cl_vol_pr = [df_1234["close"].iloc[-1], df_1234["volume"].iloc[-1], df_1234["pr_change"].iloc[-1]]
        # print(_cl_vol_pr)

        _data[symbol] = {}
        _data[symbol]["nn"] = _data_nn.copy()
        _data[symbol]["cvp"] = _cl_vol_pr.copy()

        _df_all[symbol] = df_1234.copy()

    _index = functions.get_index_value(indexes=['MOEXBMI', ], _data=_data, _df=_df_all, _percents=_percents)
    print(_index)

    print(_percents)

    functions.save_index_value_to_file(_index=_index)
