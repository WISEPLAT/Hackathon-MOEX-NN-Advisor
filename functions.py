import datetime
import math
import os
import sys
import subprocess

import numpy as np
import pandas as pd
from keras.models import load_model


def get_timeframe_moex(tf, rv=False):
    """Функция получения типа таймфрейма в зависимости от направления"""
    # - целое число 1 (1 минута), 10 (10 минут), 60 (1 час), 24 (1 день), 7 (1 неделя), 31 (1 месяц) или 4 (1 квартал)
    tfs = {"M1": 1, "M10": 10, "H1": 60, "D1": 24, "W1": 7, "MN1": 31, "Q1": 4}
    if rv: tfs = {1: "M1", 10: "M10", 60: "H1", 24: "D1", 7: "W1", 31: "MN1", 4: "Q1"}  # наоборот, если нужно )
    if tf in tfs: return tfs[tf]
    return False


def get_future_key(key, tf, future_tf):
    """Высчитываем следующий key для старшего ТФ, кроме tf == D1, W1, MN1 и кроме future_tf == W1, MN1"""
    if tf in ["D1", "W1", "MN1"] or future_tf in ["W1", "MN1"]: return False

    _hour = key.hour
    _minute = key.minute
    if future_tf not in ["D1", "W1", "MN1"]:
        future_key = datetime.datetime.fromisoformat(key.strftime('%Y-%m-%d') + f" {key.hour:02d}:00")
    else:
        future_key = datetime.datetime.fromisoformat(key.strftime('%Y-%m-%d') + " 00:00")

    tfs = {'M1': 1, 'M2': 2, 'M5': 5, 'M10': 10, 'M15': 15, 'M30': 30, 'H1': 60, 'H2': 120, 'H4': 240, 'D1': 1440, 'W1': False, 'MN1': False}

    _k = tfs[tf]
    _k2 = tfs[future_tf]
    _i1 = _minute // _k2

    future_key = future_key + datetime.timedelta(minutes=_k2 * (_i1 + 1))
    future_key2 = future_key + datetime.timedelta(minutes=_k2 * (_i1 + 1))
    # print(key, "=>", future_key, "=>", future_key2, f"\t{tf} => {future_tf}")

    # print("\t", _hour, _minute, _k, _k2, _i1)
    return key, future_key, future_key2


def detect_class(key, future_key, future_key2, arr_OHLCV_1, timeframe_1, expected_change):
    """определяем класс к которому относятся future свечи на future_key, future_key2  """
    if future_key in arr_OHLCV_1:
        _future_ohlcv = arr_OHLCV_1[future_key]
        # print(_future_ohlcv, "111111", key, "=>", future_key)
    else:
        # ищем ближайший future_key
        for k in list(arr_OHLCV_1.keys()):
            if k > key:
                future_key = k
                break
        _future_ohlcv = arr_OHLCV_1[future_key]
        # print(_future_ohlcv, "22222", key, "=>", future_key)

    # print(_future_ohlcv, "*******")
    _percent_OC = _future_ohlcv[5]  # 5 == _percent_OC
    _sign = math.copysign(1, _percent_OC)  # берем знак процента
    # print(_percent_OC, _sign)
    _classification_percent = _sign * get_classification(abs(_percent_OC), tf=timeframe_1, ex_ch=expected_change)

    if _classification_percent == 0:
        # попытка ещё раз сделать классификацию, заглянув на 2 свечи вперед
        _pre_percent_OC = _percent_OC  # учитываем % и предыдущей свечи
        future_key = future_key2
        if future_key in arr_OHLCV_1:
            _future_ohlcv = arr_OHLCV_1[future_key]
            # print(_future_ohlcv, "33333", key, "=>", future_key)
        else:
            # ищем ближайший future_key
            for k in list(arr_OHLCV_1.keys()):
                if k > key:
                    future_key = k
                    break
            _future_ohlcv = arr_OHLCV_1[future_key]
            # print(_future_ohlcv, "44444", key, "=>", future_key)
        # print(_future_ohlcv, "**222**")
        _percent_OC = _future_ohlcv[5]  # 5 == _percent_OC
        _sign = math.copysign(1, _percent_OC)  # берем знак процента
        # print(_percent_OC, _sign)
        _percent_OC += _pre_percent_OC  # учитываем % и предыдущей свечи
        _classification_percent = _sign * get_classification(abs(_percent_OC), tf=timeframe_1, ex_ch=expected_change)

    return _classification_percent


def get_classification(_p, tf, ex_ch):
    """Определяем класс по проценту свечи"""
    _class_percent = 6
    for i in range(len(ex_ch[tf]) - 1):
        if ex_ch[tf][i] <= _p < ex_ch[tf][i + 1]:
            _class_percent = i
            break
    return _class_percent


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_error_and_exit(_error, _error_code):
    '''Функция вывода ошибки и остановки программы'''
    print(bcolors.FAIL+_error+bcolors.ENDC)
    exit(_error_code)


def print_warning(_warning):
    '''Функция вывода предупреждения'''
    print(bcolors.WARNING+_warning+bcolors.ENDC)


def join_paths(paths):
    """Функция формирует путь из списка"""
    _folder = ''
    for _path in paths:
        _folder = os.path.join(_folder, _path)
    return _folder


def create_some_folders(timeframes, classes=None):
    """Функция создания необходимых директорий"""
    folder = 'NN_winner'
    if not os.path.exists(folder): os.makedirs(folder)

    folder = 'NN_winner_one'
    if not os.path.exists(folder): os.makedirs(folder)

    folder = 'csv'
    if not os.path.exists(folder): os.makedirs(folder)

    folder = 'NN'
    if not os.path.exists(folder): os.makedirs(folder)

    _folder = os.path.join(folder, f"_models")
    if not os.path.exists(_folder): os.makedirs(_folder)


def start_redirect_output_from_screen_to_file(redirect, filename):
    '''Функция старта перенаправления вывода с консоли в файл'''
    if redirect:
        sys.stdout = open(filename, 'w', encoding='utf8')


def stop_redirect_output_from_screen_to_file():
    '''Функция прекращения перенаправления вывода с консоли в файл'''
    try:
        sys.stdout.close()
    except:
        pass


def load_metric(symbol, metric):
    file_name = os.path.join('csv', f'{symbol}_{metric}.csv')
    _df = pd.DataFrame()
    try:
        _df = pd.read_csv(file_name, sep=',', parse_dates=['datetime'])  # Считываем файл в DataFrame
    except:
        pass
    return _df


def sigmoid3(x): return x / math.sqrt(1+x*x)


def start_adviser_server():
    '''Запуск рекомендательного сервера для инвесторов'''
    # print(os.getcwd())
    os.chdir('Adviser4Investors ')
    process = subprocess.Popen([sys.executable, 'run_server.py'], close_fds=True)  # https://stackoverflow.com/questions/16472337/how-to-run-a-python-file-as-thread
    os.chdir('..')
    return process


def stop_adviser_server(process):
    '''Остановка рекомендательного сервера для инвесторов'''
    process.send_signal(0)


def get_index_value(indexes, _data, _df, _percents):
    """Расчет индекса и предсказания нейросети"""
    filepath = "indexes"

    _d_arr = []
    for _index in indexes:
        df = pd.read_csv(os.path.join(filepath, f"index_{_index}_table.csv"), encoding="utf-8", sep=';')
        print(_index)
        print(df)

        _d0 = {"name": _index}
        _d0_arr = []
        for _ind, _row in df.iterrows():
            id = _row['id']
            ticker_rus = _row['ticker_rus']
            ticker = _row['ticker']
            percent = _row['percent']
            last_price = _row['last_price']
            percent_change = _row['percent_change']
            vol_mln_rub = _row['vol_mln_rub']
            cap_mlrd_rub = _row['cap_mlrd_rub']
            vol_change = _row['vol_change']

            predict = 0
            try:
                cur_run_folder = os.path.abspath(os.getcwd())  # текущий каталог
                # загружаем выбранную нами обученную нейросеть
                model = load_model(join_paths([cur_run_folder, "NN_winner", f"{ticker}_model.hdf5"]))

                _array = []
                size_of_data = 16
                for index, row in _df[ticker].iloc[-size_of_data: len(_df[ticker])].iterrows():
                    pr_change, put_vol_b, imbalance_vol, d_close = row['pr_change'], row['put_vol_b'], row['imbalance_vol'], row['d_close']
                    # print(index, pr_change, put_vol_b, imbalance_vol, d_close)
                    _array.append([pr_change, put_vol_b, imbalance_vol])

                _array = np.array(_array)
                _array = np.expand_dims(_array, axis=0)
                # print(_array.shape)

                predict = model.predict(_array, verbose=0)
                predict = predict.tolist()[0][0]
                print('_predict', predict)
            except:
                pass

            # меняем на свои
            try:
                # vol_mln_rub = _data[ticker]["cvp"][1]
                last_price = _data[ticker]["cvp"][0]
                # percent_change = _data[ticker]["cvp"][2].tolist()
                percent_change = _percents[ticker]
                # print(55555555, _data[ticker]["cvp"][2], percent_change, _data[ticker]["cvp"][2].tolist())
            except:
                pass

            try:
                percent_change = float(percent_change.replace("%", ""))
            except:
                pass

            print(id, ticker_rus, ticker, percent, last_price, percent_change, vol_mln_rub, cap_mlrd_rub, vol_change)
            _d0_arr.append({
                "name": ticker,
                "name_rus": ticker_rus,
                "volume": float(str(vol_mln_rub).replace(" ", "")),
                "value": 0,
                "price": last_price,
                "pc": "{:.2f}".format(percent_change),
                "predict": predict})

        _d0["children"] = _d0_arr
        _d_arr.append(_d0)

    _data = {
        "name": "Фондовые индексы Московской Биржи",
        "children": _d_arr
    }

    return _data


def save_index_value_to_file(_index):
    """Сохраняем индекс для отображения в файл"""
    cur_run_folder = os.path.abspath(os.getcwd())  # текущий каталог
    filename = join_paths([cur_run_folder, "web", "static", "js", "index3.js"])
    # let data = {'name': 'Фондовые индексы Московской Биржи', 'children': [{'name': 'MOEXBMI', 'children': [{'name': 'LKOH', 'name_rus': 'ЛУКОЙЛ', 'volume': 1125.99, 'value': 0, 'price': 7320.0, 'pc': '0.34', 'predict': 0}, {'name': 'SBER', 'name_rus': 'Сбербанк', 'volume': 4153.33, 'value': 0, 'price': 264.32, 'pc': '0.12', 'predict': 0}, {'name': 'GAZP', 'name_rus': 'ГАЗПРОМ ао', 'volume': 1063.65, 'value': 0, 'price': 164.25, 'pc': '-0.28', 'predict': 0}, {'name': 'GMKN', 'name_rus': 'ГМКНорНик', 'volume': 336.38, 'value': 0, 'price': 17194.0, 'pc': '0.57', 'predict': 0}, {'name': 'TATN', 'name_rus': 'Татнфт 3ао', 'volume': 453.29, 'value': 0, 'price': 646.4, 'pc': '-0.25', 'predict': 0}, {'name': 'SNGS', 'name_rus': 'Сургнфгз', 'volume': 52.76, 'value': 0, 'price': 31.965, 'pc': '-0.30', 'predict': 0}, {'name': 'NVTK', 'name_rus': 'Новатэк ао', 'volume': 155.03, 'value': 0, 'price': 1515.0, 'pc': '0.07', 'predict': 0}, {'name': 'SBER', 'name_rus': 'Сбербанк-п', 'volume': 290.51, 'value': 0, 'price': 264.32, 'pc': '0.12', 'predict': 0}, {'name': 'SNGS', 'name_rus': 'Сургнфгз-п', 'volume': 189.9, 'value': 0, 'price': 58.475, 'pc': '-0.11', 'predict': 0}, {'name': 'PLZL', 'name_rus': 'Полюс', 'volume': 85.86, 'value': 0, 'price': 11203.0, 'pc': '-0.02', 'predict': 0}, {'name': 'ROSN', 'name_rus': 'Роснефть', 'volume': 348.87, 'value': 0, 'price': 582.7, 'pc': '0.67', 'predict': 0}, {'name': 'RUAL', 'name_rus': 'РУСАЛ ао', 'volume': 113.84, 'value': 0, 'price': 37.85, 'pc': '-0.33', 'predict': 0}, {'name': 'PIKK', 'name_rus': 'ПИК ао', 'volume': 32.65, 'value': 0, 'price': 704.5, 'pc': '-0.18', 'predict': 0}, {'name': 'CHMF', 'name_rus': 'СевСт-ао', 'volume': 231.19, 'value': 0, 'price': 1305.6, 'pc': '-0.88', 'predict': 0}, {'name': 'MGNT', 'name_rus': 'Магнит ао', 'volume': 1830.2, 'value': 0, 'price': 6316.5, 'pc': '-0.10', 'predict': 0}, {'name': 'IRAO', 'name_rus': 'ИнтерРАОао', 'volume': 62.01, 'value': 0, 'price': 4.238, 'pc': '0.21', 'predict': 0}, {'name': 'NLMK', 'name_rus': 'НЛМК ао', 'volume': 496.43, 'value': 0, 'price': 178.76, 'pc': '-1.88', 'predict': 0}, {'name': 'YNDX', 'name_rus': 'Yandex clA', 'volume': 566.98, 'value': 0, 'price': 2623.8, 'pc': '1.13', 'predict': 0}, {'name': 'MAGN', 'name_rus': 'ММК', 'volume': 90.08, 'value': 0, 'price': 51.87, 'pc': '0.46', 'predict': 0}, {'name': 'TATN', 'name_rus': 'Татнфт 3ап', 'volume': 80.24, 'value': 0, 'price': 645.7, 'pc': '-0.34', 'predict': 0}, {'name': 'ALRS', 'name_rus': 'АЛРОСА ао', 'volume': 108.8, 'value': 0, 'price': 66.22, 'pc': '0.32', 'predict': 0}, {'name': 'MTSS', 'name_rus': 'МТС-ао', 'volume': 169.77, 'value': 0, 'price': 262.05, 'pc': '-0.30', 'predict': 0}, {'name': 'MOEX', 'name_rus': 'МосБиржа', 'volume': 594.21, 'value': 0, 'price': 207.92, 'pc': '1.24', 'predict': 0}, {'name': 'VTBR', 'name_rus': 'ВТБ ао', 'volume': 314.07, 'value': 0, 'price': 0.02483, 'pc': '-0.06', 'predict': 0}, {'name': 'RTKM', 'name_rus': 'Ростел -ао', 'volume': 95.2, 'value': 0, 'price': 81.25, 'pc': '0.46', 'predict': 0}, {'name': 'AGRO', 'name_rus': 'AGRO-гдр', 'volume': 85.12, 'value': 0, 'price': 1486.8, 'pc': '-0.92', 'predict': 0}, {'name': 'KAZT', 'name_rus': 'Куйбазот', 'volume': 18.76, 'value': 0, 'price': 707.6, 'pc': '2.02', 'predict': 0}, {'name': 'OZON', 'name_rus': 'OZON-адр', 'volume': 192.86, 'value': 0, 'price': 2867.0, 'pc': '-0.97', 'predict': 0}, {'name': 'PHOR', 'name_rus': 'ФосАгро ао', 'volume': 36.63, 'value': 0, 'price': 6736.0, 'pc': '-0.28', 'predict': 0}, {'name': 'ENPG', 'name_rus': 'ЭН+ГРУП ао', 'volume': 35.87, 'value': 0, 'price': 470.2, 'pc': '-1.43', 'predict': 0}, {'name': 'TCSG', 'name_rus': 'TCS-гдр', 'volume': 244.19, 'value': 0, 'price': 3446.5, 'pc': '-0.65', 'predict': 0}, {'name': 'CBOM', 'name_rus': 'МКБ ао', 'volume': 26.01, 'value': 0, 'price': 7.654, 'pc': '-0.14', 'predict': 0}, {'name': 'FLOT', 'name_rus': 'Совкомфлот', 'volume': 558.58, 'value': 0, 'price': 131.13, 'pc': '1.56', 'predict': 0}, {'name': 'TRNFP', 'name_rus': 'Транснф ап', 'volume': 347.53, 'value': 0, 'price': 146400.0, 'pc': '0.72', 'predict': 0}, {'name': 'MTLR', 'name_rus': 'Мечел ао', 'volume': 2982.82, 'value': 0, 'price': 301.63, 'pc': '-2.56', 'predict': 0}, {'name': 'VSMO', 'name_rus': 'ВСМПО-АВСМ', 'volume': 35.05, 'value': 0, 'price': 34060.0, 'pc': '-4.06', 'predict': 0}, {'name': 'AFKS', 'name_rus': 'Система ао', 'volume': 53.24, 'value': 0, 'price': 16.647, 'pc': '-0.17', 'predict': 0}, {'name': 'AFLT', 'name_rus': 'Аэрофлот', 'volume': 45.7, 'value': 0, 'price': 38.29, 'pc': '-0.57', 'predict': 0}, {'name': 'AKRN', 'name_rus': 'Акрон', 'volume': 2.86, 'value': 0, 'price': 18830.0, 'pc': '-0.53', 'predict': 0}, {'name': 'FEES', 'name_rus': 'Россети', 'volume': 189.69, 'value': 0, 'price': 0.1164, 'pc': '-1.34', 'predict': 0}, {'name': 'KZOS', 'name_rus': 'ОргСинт ао', 'volume': 15.0, 'value': 0, 'price': 107.8, 'pc': '-3.84', 'predict': 0}, {'name': 'BANE', 'name_rus': 'Башнефт ап', 'volume': 182.37, 'value': 0, 'price': 1765.5, 'pc': '1.00', 'predict': 0}, {'name': 'FIVE', 'name_rus': 'FIVE-гдр', 'volume': 61.32, 'value': 0, 'price': 2174.0, 'pc': '-0.23', 'predict': 0}, {'name': 'MTLR', 'name_rus': 'Мечел ап', 'volume': 448.12, 'value': 0, 'price': 359.1, 'pc': '-1.52', 'predict': 0}, {'name': 'SMLT', 'name_rus': 'Самолет ао', 'volume': 77.54, 'value': 0, 'price': 3986.5, 'pc': '-0.15', 'predict': 0}, {'name': 'VKCO', 'name_rus': 'МКПАО "ВК"', 'volume': 374.68, 'value': 0, 'price': 618.2, 'pc': '-1.28', 'predict': 0}, {'name': 'FESH', 'name_rus': 'ДВМП ао', 'volume': 260.21, 'value': 0, 'price': 80.97, 'pc': '-3.41', 'predict': 0}, {'name': 'NMTP', 'name_rus': 'НМТП ао', 'volume': 119.91, 'value': 0, 'price': 11.055, 'pc': '-2.17', 'predict': 0}, {'name': 'MSNG', 'name_rus': '+МосЭнерго', 'volume': 9.91, 'value': 0, 'price': 3.0535, 'pc': '0.66', 'predict': 0}, {'name': 'SELG', 'name_rus': 'Селигдар', 'volume': 117.04, 'value': 0, 'price': 68.95, 'pc': '-2.30', 'predict': 0}, {'name': 'POSI', 'name_rus': 'iПозитив', 'volume': 49.62, 'value': 0, 'price': 2143.4, 'pc': '-0.20', 'predict': 0}, {'name': 'UPRO', 'name_rus': 'Юнипро ао', 'volume': 91.38, 'value': 0, 'price': 2.147, 'pc': '-1.24', 'predict': 0}, {'name': 'QIWI', 'name_rus': 'iQIWI', 'volume': 6.97, 'value': 0, 'price': 545.5, 'pc': '-1.27', 'predict': 0}, {'name': 'BSPB', 'name_rus': 'БСП ао', 'volume': 232.32, 'value': 0, 'price': 237.51, 'pc': '-1.86', 'predict': 0}, {'name': 'HYDR', 'name_rus': 'РусГидро', 'volume': 175.84, 'value': 0, 'price': 0.8126, 'pc': '2.39', 'predict': 0}, {'name': 'NKNC', 'name_rus': 'НКНХ ао', 'volume': 26.64, 'value': 0, 'price': 106.7, 'pc': '-5.99', 'predict': 0}, {'name': 'POLY', 'name_rus': 'Polymetal', 'volume': 1131.1, 'value': 0, 'price': 422.8, 'pc': '6.90', 'predict': 0}, {'name': 'SGZH', 'name_rus': 'Сегежа', 'volume': 438.61, 'value': 0, 'price': 3.908, 'pc': '-0.81', 'predict': 0}, {'name': 'LENT', 'name_rus': 'Лента ао', 'volume': 2.18, 'value': 0, 'price': 712.5, 'pc': '-0.07', 'predict': 0}, {'name': 'GLTR', 'name_rus': 'GLTR-гдр', 'volume': 39.63, 'value': 0, 'price': 620.65, 'pc': '0.40', 'predict': 0}, {'name': 'RASP', 'name_rus': 'Распадская', 'volume': 206.18, 'value': 0, 'price': 373.7, 'pc': '-1.66', 'predict': 0}, {'name': 'LSNG', 'name_rus': 'РСетиЛЭ-п', 'volume': 5.87, 'value': 0, 'price': 196.3, 'pc': '-0.51', 'predict': 0}, {'name': 'RENI', 'name_rus': 'Ренессанс', 'volume': 54.01, 'value': 0, 'price': 95.64, 'pc': '-1.40', 'predict': 0}, {'name': 'RTKM', 'name_rus': 'Ростел -ап', 'volume': 119.97, 'value': 0, 'price': 76.6, 'pc': '0.72', 'predict': 0}, {'name': 'ETLN', 'name_rus': 'ETLN-гдр', 'volume': 20.87, 'value': 0, 'price': 79.5, 'pc': '-1.46', 'predict': 0}, {'name': 'HHR', 'name_rus': 'iHHRU-адр', 'volume': 14.38, 'value': 0, 'price': 3596.0, 'pc': '-0.11', 'predict': 0}, {'name': 'NKNC', 'name_rus': 'НКНХ ап', 'volume': 20.69, 'value': 0, 'price': 77.0, 'pc': '-2.70', 'predict': 0}, {'name': 'GEMC', 'name_rus': 'GEMC-гдр', 'volume': 5.27, 'value': 0, 'price': 834.8, 'pc': '-0.74', 'predict': 0}, {'name': 'AQUA', 'name_rus': 'ИНАРКТИКА', 'volume': 32.18, 'value': 0, 'price': 944.0, 'pc': '0.05', 'predict': 0}, {'name': 'LSRG', 'name_rus': 'ЛСР ао', 'volume': 12.51, 'value': 0, 'price': 662.4, 'pc': '-1.46', 'predict': 0}, {'name': 'NKHP', 'name_rus': 'НКХП ао', 'volume': 78.67, 'value': 0, 'price': 952.0, 'pc': '-4.80', 'predict': 0}, {'name': 'FIXP', 'name_rus': 'FIXP-гдр', 'volume': 111.71, 'value': 0, 'price': 312.5, 'pc': '-2.86', 'predict': 0}, {'name': 'OGKB', 'name_rus': 'ОГК-2 ао', 'volume': 35.9, 'value': 0, 'price': 0.543, 'pc': '-2.11', 'predict': 0}, {'name': 'MGTS', 'name_rus': 'МГТС-4ап', 'volume': 1.9, 'value': 0, 'price': 1526.0, 'pc': '-0.91', 'predict': 0}, {'name': 'APTK', 'name_rus': 'Аптеки36и6', 'volume': 14.77, 'value': 0, 'price': 14.186, 'pc': '-1.28', 'predict': 0}, {'name': 'MRKP', 'name_rus': 'РСетиЦП ао', 'volume': 15.6, 'value': 0, 'price': 0.3185, 'pc': '-1.24', 'predict': 0}, {'name': 'RNFT', 'name_rus': 'РуссНфт ао', 'volume': 26.7, 'value': 0, 'price': 156.0, 'pc': '-0.13', 'predict': 0}, {'name': 'MSRS', 'name_rus': 'РСетиМР ао', 'volume': 10.41, 'value': 0, 'price': 1.2665, 'pc': '-0.86', 'predict': 0}, {'name': 'MDMG', 'name_rus': 'MDMG-гдр', 'volume': 4.31, 'value': 0, 'price': 811.9, 'pc': '0.36', 'predict': 0}, {'name': 'SFIN', 'name_rus': 'ЭсЭфАй ао', 'volume': 19.64, 'value': 0, 'price': 571.0, 'pc': '1.35', 'predict': 0}, {'name': 'TGKA', 'name_rus': 'ТГК-1', 'volume': 75.03, 'value': 0, 'price': 0.009588, 'pc': '-2.86', 'predict': 0}, {'name': 'ELFV', 'name_rus': 'ЭЛ5Энер ао', 'volume': 47.27, 'value': 0, 'price': 0.6496, 'pc': '-2.58', 'predict': 0}, {'name': 'MVID', 'name_rus': 'М.видео', 'volume': 41.78, 'value': 0, 'price': 177.3, 'pc': '-1.94', 'predict': 0}, {'name': 'MRKC', 'name_rus': 'РоссЦентр', 'volume': 15.75, 'value': 0, 'price': 0.5842, 'pc': '-1.85', 'predict': 0}, {'name': 'BELU', 'name_rus': 'НоваБев ао', 'volume': 93.27, 'value': 0, 'price': 5508.0, 'pc': '-0.31', 'predict': 0}, {'name': 'MRKU', 'name_rus': 'Россети Ур', 'volume': 5.08, 'value': 0, 'price': 0.3958, 'pc': '-1.25', 'predict': 0}, {'name': 'SVAV', 'name_rus': 'СОЛЛЕРС', 'volume': 37.15, 'value': 0, 'price': 759.0, 'pc': '-2.50', 'predict': 0}, {'name': 'CIAN', 'name_rus': 'CIAN-адр', 'volume': 36.25, 'value': 0, 'price': 672.4, 'pc': '-3.00', 'predict': 0}, {'name': 'RKKE', 'name_rus': 'ЭнергияРКК', 'volume': 10.34, 'value': 0, 'price': 23220.0, 'pc': '1.09', 'predict': 0}, {'name': 'DVEC', 'name_rus': 'ДЭК ао', 'volume': 12.43, 'value': 0, 'price': 3.27, 'pc': '-2.97', 'predict': 0}, {'name': 'WUSH', 'name_rus': 'iВУШХолднг', 'volume': 80.39, 'value': 0, 'price': 236.1, 'pc': '-0.71', 'predict': 0}, {'name': 'TGKB', 'name_rus': 'ТГК-2', 'volume': 20.9, 'value': 0, 'price': 0.01132, 'pc': '-2.96', 'predict': 0}, {'name': 'KZOS', 'name_rus': 'ОргСинт ап', 'volume': 10.86, 'value': 0, 'price': 26.3, 'pc': '-4.92', 'predict': 0}, {'name': 'MRKZ', 'name_rus': 'РСетиСЗ ао', 'volume': 10.27, 'value': 0, 'price': 0.092, 'pc': '-3.97', 'predict': 0}, {'name': 'MRKV', 'name_rus': 'РсетВол ао', 'volume': 15.84, 'value': 0, 'price': 0.0546, 'pc': '0.46', 'predict': 0}, {'name': 'CHMK', 'name_rus': 'ЧМК ао', 'volume': 1.76, 'value': 0, 'price': 10340.0, 'pc': '-0.96', 'predict': 0}, {'name': 'TTLK', 'name_rus': 'Таттел. ао', 'volume': 9.09, 'value': 0, 'price': 1.061, 'pc': '-2.66', 'predict': 0}, {'name': 'SPBE', 'name_rus': 'СПБ Биржа', 'volume': 1145.74, 'value': 0, 'price': 88.0, 'pc': '-10.20', 'predict': 0}, {'name': 'ISKJ', 'name_rus': 'iАРТГЕН ао', 'volume': 6.28, 'value': 0, 'price': 96.1, 'pc': '-0.93', 'predict': 0}, {'name': 'OKEY', 'name_rus': 'OKEY-гдр', 'volume': 2.78, 'value': 0, 'price': 34.54, 'pc': '-0.55', 'predict': 0}]}]}
    with open(filename, 'w', encoding="UTF-8") as file:
        file.write(
f"""let data = {_index}""")


def get_percent(_df):
    """Подсчет изменения в % с начала дня"""
    # today = datetime.datetime.now().date().strftime('%Y-%m-%d')
    today = _df["datetime"].iloc[-1].date().strftime('%Y-%m-%d')
    today = datetime.datetime.fromisoformat(today)
    j = -1
    sum = 0
    # print(_df)
    # print(df_1["datetime"].iloc[j])
    while _df["datetime"].iloc[j] > today:
        # print(_df["datetime"].iloc[j])
        j -= 1
        sum += _df["pr_change"].iloc[j]
    return sum
