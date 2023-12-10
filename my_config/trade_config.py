# конфигурационный файл для торговой стратегии - все параметры корректируем "руками"

class Config:

    index = "MOEXBMI"  # индекс по которому будем получать рекомендации от нейросети, выбираем MOEXBMI - Индекс широкого рынка

    # тикеры по которым обучаем нейросеть, по ним же и предсказываем, берем из файла 0_get_indices_from_MOEX.py
    training_NN = ['ABIO', 'AFKS', 'AFLT', 'AGRO', 'AKRN', 'ALRS', 'APTK', 'AQUA', 'BANEP', 'BELU', 'BSPB', 'CBOM',
                   'CHMF', 'CHMK', 'CIAN', 'DVEC', 'ELFV', 'ENPG', 'ETLN', 'FEES', 'FESH', 'FIVE', 'FIXP', 'FLOT',
                   'GAZP', 'GEMC', 'GLTR', 'GMKN', 'HHRU', 'HYDR', 'IRAO', 'KAZT', 'KZOS', 'KZOSP', 'LENT', 'LKOH',
                   'LSNGP', 'LSRG', 'MAGN', 'MDMG', 'MGNT', 'MGTSP', 'MOEX', 'MRKC', 'MRKP', 'MRKU', 'MRKV', 'MRKZ',
                   'MSNG', 'MSRS', 'MTLR', 'MTLRP', 'MTSS', 'MVID', 'NKHP', 'NKNC', 'NKNCP', 'NLMK', 'NMTP', 'NVTK',
                   'OGKB', 'OKEY', 'OZON', 'PHOR', 'PIKK', 'PLZL', 'POLY', 'POSI', 'QIWI', 'RASP', 'RENI', 'RKKE',
                   'RNFT', 'ROSN', 'RTKM', 'RTKMP', 'RUAL', 'SBER', 'SBERP', 'SELG', 'SFIN', 'SGZH', 'SMLT', 'SNGS',
                   'SNGSP', 'SPBE', 'SVAV', 'TATN', 'TATNP', 'TCSG', 'TGKA', 'TGKB', 'TRNFP', 'TTLK', 'UPRO', 'VKCO',
                   'VSMO', 'VTBR', 'WUSH', 'YNDX']

    timeframe_0 = "M5"  # ТФ для обучения нейросети - вход, на нём же и будем делать прогнозы - данные из Super Candles
    timeframe_1 = "M10"  # ТФ для обучения нейросети - выход - данные из Candles
    start = "2020-01-01"  # с какой даты загружаем исторические данные с MOEX

    trading_hours_start = "10:00"  # время работы биржи - начало
    trading_hours_end = "18:50"  # время работы биржи - конец

