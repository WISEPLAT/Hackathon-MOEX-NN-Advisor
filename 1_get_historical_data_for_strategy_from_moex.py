"""
    В этом коде мы получаем исторические данные с MOEX
    и сохраняем их в CSV файлы.
    Авторы: Олег Шпагин, Федор Шпагин
    Github: https://github.com/WISEPLAT
    Telegram: https://t.me/OlegSh777
"""

exit(777)  # для запрета запуска кода, иначе перепишет результаты

# pip install moexalgo pandas

import functions
import functions_algopack as al  # используем moexalgo для получения данных

from time import time
from datetime import datetime, timedelta
from my_config.trade_config import Config  # Файл конфигурации


if __name__ == "__main__":

    start_time = time()  # Время начала запуска скрипта

    # применение настроек из config.py
    training_NN = Config.training_NN  # тикеры по которым скачиваем исторические данные
    timeframe_0 = Config.timeframe_0  # таймфрейм для обучения нейросети - вход - данные из Super Candles
    timeframe_1 = Config.timeframe_1  # таймфрейм для обучения нейросети - выход - данные из Candles
    start = datetime.strptime(Config.start, "%Y-%m-%d")  # с какой даты загружаем исторические данные с MOEX
    end = datetime.now()  # Получать данные будем до текущей даты

    # создаем необходимые каталоги
    functions.create_some_folders(timeframes=[timeframe_0, timeframe_1])

    # training_NN = ['ABIO', ]  # для теста ТФ1 -> ТФ5 -> ТФ10

    metrics = ('tradestats', 'orderstats', 'obstats')  # Метрики
    # metrics = ('tradestats', )  # будем использовать некоторые метрики из tradestats, у них только один ТФ = M5
    al.save_metrics_to_files(training_NN, metrics, folder="csv", _from=start, _to=end)  # Сохраняем метрики тикеров в файлы

    # time_frames = ('Q', 'M', 'W', 'D', 60, 10, 1)  # Все временнЫе интервалы
    time_frames = (10, )  # ВременнЫе интервалы
    # time_frames = (1, )  # ВременнЫе интервалы
    al.save_candles_to_files(training_NN, time_frames, folder="csv", _from=start, _to=end)  # Сохраняем бары тикеров в файлы

    print(f'Скрипт выполнен за {(time() - start_time):.2f} с')
