from datetime import datetime
from time import time
import os.path

import pandas as pd

from moexalgo import Ticker


from_date = datetime(2020, 1, 1)  # Получать данные будем с первой возможной даты и времени Алгопака
till_date = datetime.now()  # Получать данные будем до текущей даты
limit = 10_000  # Какое кол-во записей будем получать в каждом запросе. Максимум 50_000


def save_candles_to_files(symbols=('SBER',), time_frames=('D',),
                          skip_first_date=False, skip_last_date=False, four_price_doji=False,
                          # date_format='%d.%m.%Y %H:%M', sep='\t', decimal=".", ext="txt",
                          date_format="%Y-%m-%d %H:%M", sep=",", decimal=".", ext="csv", folder="",
                          _from=from_date, _to=till_date):
    """Сохранение баров тикеров по временнЫм интервалам в файлы
        :param list symbols: Коды тикеров в виде кортежа
        :param tuple time_frames: ВременнЫе интервалы в виде кортежа. В минутах (int) 1, 10, 60 или код ("D" - дни, "W" - недели, "M" - месяцы, "Q" - кварталы)
        :param bool skip_first_date: Убрать бары на первую полученную дату
        :param bool skip_last_date: Убрать бары на последнюю полученную дату
        :param bool four_price_doji: Оставить бары с дожи 4-х цен
        :param str date_format: Формат поля даты для выгрузки в csv
        :param str sep: Разделитель для выгрузки в csv
        :param str decimal: Разделитель float для выгрузки в csv
        :param str folder: Путь для сохранения выгрузки csv
        :param datetime _from: С какой даты выгружаем в csv
        :param datetime _to: До какой даты выгружаем в csv
    """
    for symbol in symbols:  # Пробегаемся по всем тикерам
        for time_frame in time_frames:  # Пробегаемся по всем временнЫм интервалам
            save_candles_to_file(symbol, time_frame, skip_first_date, skip_last_date, four_price_doji,
                                 date_format=date_format, sep=sep, decimal=decimal, ext=ext, folder=folder,
                                 _from=_from, _to=_to)


def save_candles_to_file(symbol='SBER', time_frame='M',
                         skip_first_date=False, skip_last_date=False, four_price_doji=False,
                         # date_format='%d.%m.%Y %H:%M', sep='\t', decimal=".", ext="txt",
                         date_format="%Y-%m-%d %H:%M", sep=",", decimal=".", ext="csv", folder="",
                         _from=from_date, _to=till_date):
    """Получение баров, объединение с имеющимися барами в файле (если есть), сохранение баров в файл
        :param int|str time_frame: Временной интервал в минутах (int) 1, 10, 60 или код ("D" - дни, "W" - недели, "M" - месяцы, "Q" - кварталы)
        :param bool skip_first_date: Убрать бары на первую полученную дату
        :param bool skip_last_date: Убрать бары на последнюю полученную дату
        :param bool four_price_doji: Оставить бары с дожи 4-х цен
        :param str date_format: Формат поля даты для выгрузки в csv
        :param str sep: Разделитель для выгрузки в csv
        :param str decimal: Разделитель float для выгрузки в csv
        :param str folder: Путь для сохранения выгрузки csv
        :param datetime _from: С какой даты выгружаем в csv
        :param datetime _to: До какой даты выгружаем в csv
    """
    _to = _to.strftime("%Y-%m-%d")
    tf = f'{time_frame}1' if time_frame in ('D', 'W', 'Q') else f'MN1' if time_frame == 'M' else f'M{time_frame}'  # Временной интервал для файла
    file_df = pd.DataFrame()  # Дальше будем пытаться получить бары из файла
    file_name = os.path.join(folder, f'{symbol}_{tf}.{ext}')
    file_exists = os.path.isfile(file_name)  # Существует ли файл
    if file_exists:  # Если файл существует
        print(f'Получение файла {file_name}')
        file_df = pd.read_csv(file_name, sep=sep, parse_dates=['datetime'], date_format=date_format, decimal=decimal)  # Считываем файл в DataFrame
        last_dt = file_df.iloc[-1]['datetime']  # Получать данные будем с последней полученной даты и времени из файла
        print(f'- Первая запись файла   : {file_df.iloc[0]["datetime"]}')
        print(f'- Последняя запись файла: {last_dt}')
        print(f'- Кол-во записей в файле: {len(file_df)}')
    else:  # Файл не существует
        print(f'Файл {file_name} не найден и будет создан')
        last_dt = _from  # Получать данные будем с первой возможной даты и времени Алгопака
    last_date = last_dt.date()  # Получать данные будем с последней полученной даты из файла или с первой возможной даты Алгопака
    print(f'Получение истории {symbol} {tf} с ММВБ')
    ticker = Ticker(symbol)  # Тикер ММВБ
    while True:  # Будем получать данные пока не получим все
        iterator = ticker.candles(date=last_date, till_date=_to, period=time_frame, limit=limit)  # История. Максимум, 50000 баров
        rows_list = []  # Будем собирать строки в список
        for it in iterator:  # Итерируем генератор
            rows_list.append(it.__dict__)  # Класс превращаем в словарь, добавляем строку в список
        if rows_list:
            stats = pd.DataFrame(rows_list)  # Из списка создаем pandas DataFrame
            stats.rename(columns={'begin': 'datetime'}, inplace=True)  # Переименовываем колонку даты и времени
            if not file_exists and skip_first_date:  # Если файла нет, и убираем бары на первую дату
                len_with_first_date = len(stats)  # Кол-во баров до удаления на первую дату
                first_date = stats.iloc[0]['datetime'].date()  # Первая дата
                stats.drop(stats[(stats['datetime'].date() == first_date)].index, inplace=True)  # Удаляем их
                print(f'- Удалено баров на первую дату {first_date}: {len_with_first_date - len(stats)}')
            if skip_last_date:  # Если убираем бары на последнюю дату
                len_with_last_date = len(stats)  # Кол-во баров до удаления на последнюю дату
                last_date = stats.iloc[-1]['datetime'].date()  # Последняя дата
                stats.drop(stats[(stats['datetime'].date() == last_date)].index, inplace=True)  # Удаляем их
                print(f'- Удалено баров на последнюю дату {last_date}: {len_with_last_date - len(stats)}')
            if not four_price_doji:  # Если удаляем дожи 4-х цен
                len_with_doji = len(stats)  # Кол-во баров до удаления дожи
                stats.drop(stats[(stats['high'] == stats['low'])].index, inplace=True)  # Удаляем их по условия High == Low
                print('- Удалено дожи 4-х цен:', len_with_doji - len(stats))
            if len(stats) == 0:  # Если нечего объединять
                print('Новых записей нет')
                break  # то переходим к следующему тикеру, дальше не продолжаем
            last_stats_dt = stats.iloc[-1]['datetime']  # Последняя полученная дата и время
            last_stats_date = last_stats_dt.date()  # Последняя полученная дата
            if last_stats_dt == last_dt:  # Если не получили новые значения
                print('Все данные получены')
                break  # то переходим к следующему тикеру, дальше не продолжаем
            print('- Получены данные с', stats.iloc[0]['datetime'], 'по', last_stats_dt)
            file_df = pd.concat([file_df, stats]).drop_duplicates(keep='last')  # Добавляем новые данные в существующие. Удаляем дубликаты. Сбрасываем индекс
            file_df = file_df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'value']]  # Отбираем нужные колонки. Дата и время будет экспортирована как индекс
            file_df.set_index('datetime').to_csv(file_name, sep=sep, date_format=date_format, decimal=decimal)  # На каждой итерации будем сохранять результат в файл
            last_dt = last_stats_dt  # Запоминаем последние полученные дату и время
            last_date = last_stats_date  # и дату
        else:
            break


def save_metrics_to_files(symbols=('SBER',), metrics=('tradestats', 'orderstats', 'obstats'),
                          # date_format='%d.%m.%Y %H:%M', sep='\t', decimal=".", ext="txt",):
                          date_format="%Y-%m-%d %H:%M", sep=",", decimal=".", ext="csv", folder="",
                          _from=from_date, _to=till_date):
    """Сохранение метрик тикеров в файлы

    :param list symbols: Коды тикеров в виде кортежа
    :param tuple metrics: Метрики. 'tradestats' - сделки, 'orderstats' - заявки, 'obstats' - стакан
    :param str date_format: Формат поля даты для выгрузки в csv
    :param str sep: Разделитель для выгрузки в csv
    :param str decimal: Разделитель float для выгрузки в csv
    :param str folder: Путь для сохранения выгрузки csv
    :param datetime _from: С какой даты выгружаем в csv
    :param datetime _to: До какой даты выгружаем в csv
    """
    for symbol in symbols:  # Пробегаемся по всем тикерам
        for metric in metrics:  # Пробегаемся по всем метрикам
            save_metric_to_file(symbol, metric, date_format=date_format, sep=sep, decimal=decimal,
                                ext=ext, folder=folder, _from=_from, _to=_to)  # Получаем метрику тикера, сохраняем в файл


def save_metric_to_file(symbol='SBER', metric='tradestats',
                        # date_format='%d.%m.%Y %H:%M', sep='\t', decimal=".", ext="txt"):
                        date_format="%Y-%m-%d %H:%M", sep=",", decimal=".", ext="csv", folder="",
                        _from=from_date, _to=till_date):
    """Получение метрики тикера, сохранение в файл

    :param str symbol: Код тикера
    :param str metric: Метрика. 'tradestats' - сделки, 'orderstats' - заявки, 'obstats' - стакан
    :param str date_format: Формат поля даты для выгрузки в csv
    :param str sep: Разделитель для выгрузки в csv
    :param str decimal: Разделитель float для выгрузки в csv
    :param str folder: Путь для сохранения выгрузки csv
    :param datetime _from: С какой даты выгружаем в csv
    :param datetime _to: До какой даты выгружаем в csv
    """
    _to = _to.strftime("%Y-%m-%d")
    file_df = pd.DataFrame()  # Дальше будем пытаться получить бары из файла
    file_name = os.path.join(folder, f'{symbol}_{metric}.{ext}')
    file_exists = os.path.isfile(file_name)  # Существует ли файл
    if file_exists:  # Если файл существует
        print(f'Получение файла {file_name}')
        file_df = pd.read_csv(file_name, sep=sep, parse_dates=['datetime'], date_format=date_format, decimal=decimal)  # Считываем файл в DataFrame
        last_dt = file_df.iloc[-1]['datetime']  # Получать данные будем с последней полученной даты и времени из файла
        print(f'- Первая запись файла   : {file_df.iloc[0]["datetime"]}')
        print(f'- Последняя запись файла: {last_dt}')
        print(f'- Кол-во записей в файле: {len(file_df)}')
    else:  # Файл не существует
        print(f'Файл {file_name} не найден и будет создан')
        last_dt = _from  # Получать данные будем с первой возможной даты и времени Алгопака
    last_date = last_dt.date()  # Получать данные будем с последней полученной даты из файла или с первой возможной даты Алгопака
    print(f'Получение метрики {metric} {symbol} с ММВБ')
    ticker = Ticker(symbol)  # Тикер ММВБ
    while True:  # Будем получать данные пока не получим все
        if metric == 'tradestats':  # Сделки
            iterator = ticker.tradestats(date=last_date, till_date=_to, limit=limit)
        elif metric == 'orderstats':  # Заявки
            iterator = ticker.orderstats(date=last_date, till_date=_to, limit=limit)
        elif metric == 'obstats':  # Стакан
            iterator = ticker.obstats(date=last_date, till_date=_to, limit=limit)
        else:
            print('Метрика указана неверно')
            break
        rows_list = []  # Будем собирать строки в список
        for it in iterator:  # Итерируем генератор
            rows_list.append(it.__dict__)  # Класс превращаем в словарь, добавляем строку в список
        if rows_list:
            stats = pd.DataFrame(rows_list)  # Из списка создаем pandas DataFrame
            stats.drop('secid', axis='columns', inplace=True)  # Удаляем колонку тикера. Название тикера есть в имени файла
            stats.rename(columns={'ts': 'datetime'}, inplace=True)  # Переименовываем колонку даты и времени
            stats['datetime'] -= pd.Timedelta(minutes=5)  # 5-и минутку с датой и временем окончания переводим в дату и время начала для синхронизации с OHLCV
            last_stats_dt = stats.iloc[-1]['datetime']  # Последняя полученная дата и время
            last_stats_date = last_stats_dt.date()  # Последняя полученная дата
            if last_stats_dt == last_dt:  # Если не получили новые значения
                print('Все данные получены')
                break  # то выходим, дальше не продолжаем
            print('- Получены данные с', stats.iloc[0]['datetime'], 'по', last_stats_dt)
            file_df = pd.concat([file_df, stats]).drop_duplicates(keep='last')  # Добавляем новые данные в существующие. Удаляем дубликаты. Сбрасываем индекс
            file_df.set_index('datetime').to_csv(file_name, sep=sep, date_format=date_format, decimal=decimal)  # На каждой итерации будем сохранять результат в файл
            last_dt = last_stats_dt  # Запоминаем последние полученные дату и время
            last_date = last_stats_date  # и дату
        else:
            break
