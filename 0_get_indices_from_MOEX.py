"""
    В этом коде мы получаем данные с MOEX по индексам
    и выводим их на экран.

    Авторы: Олег Шпагин, Федор Шпагин
    Github: https://github.com/WISEPLAT
    Telegram: https://t.me/OlegSh777
"""

import asyncio
import requests

import apimoex
import pandas as pd

from my_config.trade_config import Config  # Файл конфигурации торгового робота


pd.options.display.width = None
pd.set_option('display.max_rows', 300)


async def main():
    arguments = {}
    api_url = 'https://iss.moex.com/iss/statistics/engines/stock/markets/index/analytics.json'  # Индексы фондового рынка
    print(f"Список индексов:")

    with requests.Session() as session:
        iss = apimoex.ISSClient(session, api_url, arguments)
        # data = iss.get_all()
        data = iss.get()
        # print(data)
        for _key in data.keys():
            df = pd.DataFrame(data[_key]) # Вес тикеров в индексе
            # print(_key)
            print(df)

    ind = Config.index
    api_url = f'https://iss.moex.com/iss/statistics/engines/stock/markets/index/analytics/{ind}.json'  # инфо по индексу
    print(f"\nИндекс {ind}:")

    df = None
    with requests.Session() as session:
        iss = apimoex.ISSClient(session, api_url)
        data = iss.get_all()
        df = pd.DataFrame(data['analytics'])
        print(df)

    index_tickers = df["ticker"].tolist()
    print(f"\nТикеры входящие в индекс {ind}: {index_tickers}")


asyncio.run(main())
