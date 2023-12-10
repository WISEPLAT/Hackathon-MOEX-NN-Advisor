"""
    В этом коде реализована проверка предсказаний нейросетью для одного тикера,
    модель берется из папки NN_winner_one.

    Авторы: Олег Шпагин, Федор Шпагин
    Github: https://github.com/WISEPLAT
    Telegram: https://t.me/OlegSh777
"""

import os
import functions
import numpy as np
import pandas as pd

from keras.models import load_model

from my_config.trade_config import Config  # Файл конфигурации


if __name__ == "__main__":

    timeframe_0 = Config.timeframe_0  # таймфрейм на котором торгуем == таймфрейму на котором обучали нейросеть

    symbol = 'SBER'  # Тикер, который будем исследовать

    cur_run_folder = os.path.abspath(os.getcwd())  # текущий каталог

    # загружаем выбранную нами обученную нейросеть
    model = load_model(functions.join_paths([cur_run_folder, "NN_winner_one", f"{symbol}_model.hdf5"]))
    # Проверяем её архитектуру
    print(model.summary())

    # train accuracy = 95.7535%
    # test accuracy = 96.1988%
    # test error = 757 out of 19915 examples
    #
    # Model: "sequential_19"
    # _________________________________________________________________
    #  Layer (type)                Output Shape              Param #
    # =================================================================
    #  lstm_38 (LSTM)              (None, 16, 17)            1428
    #
    #  batch_normalization_19 (Bat  (None, 16, 17)           68
    #  chNormalization)
    #
    #  lstm_39 (LSTM)              (None, 3)                 252
    #
    #  dropout_19 (Dropout)        (None, 3)                 0
    #
    #  dense_19 (Dense)            (None, 1)                 4
    #
    # =================================================================
    # Total params: 1,752
    # Trainable params: 1,718
    # Non-trainable params: 34
    # _________________________________________________________________

    # загружаем данные для теста предсказания
    df_1 = functions.load_metric(symbol=symbol, metric='tradestats')
    df_2 = functions.load_metric(symbol=symbol, metric='orderstats')
    df_3 = functions.load_metric(symbol=symbol, metric='obstats')

    df_1.set_index('datetime', inplace=True)
    df_2.set_index('datetime', inplace=True)
    df_3.set_index('datetime', inplace=True)

    df_123 = pd.concat([df_1, df_2, df_3], axis=1)

    print(df_123)

    df_nn = df_123.copy()[["pr_close", "pr_change", "put_vol_b", "imbalance_vol"]]
    df_nn["d_close"] = df_nn["pr_close"].diff()  # чтобы смотреть закрытие выше предыдущего или ниже
    df_nn.dropna(inplace=True)

    print(df_nn)

    df_nn['pr_change'] = df_nn['pr_change'].apply(functions.sigmoid3)
    df_nn['put_vol_b'] = df_nn['put_vol_b'].diff()
    df_nn['put_vol_b'] = df_nn['put_vol_b'].apply(functions.sigmoid3)
    df_nn['imbalance_vol'] = df_nn['imbalance_vol'].apply(functions.sigmoid3)

    df_nn["d_close"] = np.where(df_nn['d_close'] > 0, 1, 0)

    _ = df_nn.pop("pr_close")

    # пробуем удалить лишнее
    # _ = df_nn.pop("put_vol_b")

    df_nn.dropna(inplace=True)

    print(df_nn)

    print(df_nn.describe().transpose())

    size_of_data = 16

    # print(df_nn.tail(size_of_data))
    print(df_nn.iloc[-size_of_data:])
    print("*" * 30)

    # сделаем N предсказаний
    N = 5
    for j in range(N):
        _array = []
        _last_d_close = 0
        for index, row in df_nn.iloc[-size_of_data - N + j + 1:len(df_nn) - N + j + 1].iterrows():
            pr_change, put_vol_b, imbalance_vol, d_close = row['pr_change'], row['put_vol_b'], row['imbalance_vol'], row['d_close']
            # print(index, pr_change, put_vol_b, imbalance_vol, d_close)
            _array.append([pr_change, put_vol_b, imbalance_vol])
            _last_d_close = d_close

        print(df_nn.iloc[-size_of_data - N + j + 1:len(df_nn) - N + j + 1])
        _array = np.array(_array)
        _array = np.expand_dims(_array, axis=0)
        print(_array.shape)

        _predict = model.predict(_array, verbose=0)
        print("_predict = ", _predict, _predict.tolist()[0][0], " ==>", _last_d_close)

        # сравниваем показатели
        if _predict.tolist()[0][0] >= 0.5 and _last_d_close == 1:
            print("Предсказано верно.")
        elif _predict.tolist()[0][0] < 0.5 and _last_d_close == 0:
            print("Предсказано верно.")
        else:
            print("Предсказано НЕ верно.")
