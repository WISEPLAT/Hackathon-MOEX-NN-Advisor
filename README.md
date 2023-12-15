# Торговый советник для инвесторов с использованием нейросетей
## Hackathon-MOEX-NN-Advisor

```
Сделан в рамках соревнования Хакатон [GO ALGO] организатором которого выступает биржа MOEX.
Было поставлено две задачи:
1) Для себя - исследовать данные, провести backtest, разработать свой торговый алгоритм и реализовать 
2) Разработать решение для инвесторов на рынке акций
Данные, сигналы и аналитика для алгоритмической торговли нужно брать через API AlgoPack. 
```

## Какую задачу мы решили перед созданием торгового советника?

Чтобы упростить жизнь не только себе, но и получить отличный инструмент для:
  - исследования данных 
  - создания и тестирования торговых стратегий
  - возможности интуитивной разработки торговой стратегии
  - создания своих индикаторов или использования уже 100+ готовых индикаторов
  - работы в live режиме с брокерами - с возможностью выставлением заявок на покупку/продажу и др.

мы разработали библиотеку [**backtrader_moexalgo**](https://github.com/WISEPLAT/backtrader_moexalgo), которая интегрирует API AlgoPack с гибкой и популярной системой создания торговых роботов
и получилась такая связка [MOEX API AlgoPack](https://www.moex.com/ru/algopack/about) + [Backtrader](https://github.com/WISEPLAT/backtrader ):
  - разместив в ней множество примеров по работе с данными и SUPER CANDLES
  - особо интересна возможность работы в live режиме, позволяющая делать полноценных торговых роботов
  - конечно, помним, что перед выходом в Live - все сначала тестируем в Offline в реализованной возможности backtesting
    - есть очень много документации 

Хотелось бы дополнительно отметить, что такие библиотеки сильно упрощают сложность разработки торговых роботов для новых разработчиков.
Тем самым происходит большая популяризация алготрейдинга. 

**P.S. Тем более, когда есть столько рабочих примеров с объяснениями, что делать и как!** 

### Почему мы выбрали использование нейросетей для торгового советника?
1. Тема использования искусственного интеллекта актуальна:
   - для прогнозирования поведения фондового рынка в целом, 
   - для осуществления предсказаний поведения цены отдельных акций и/или фьючерсов и других инструментов
   - для поиска определенных торговых формаций на графиках цен
   - API AlgoPack предоставляет множество расширенных данных, которые можно анализировать и искать ПЛЮСЫ в их использовании
   

2. Широкое применение искусственного интеллекта очень активно развивается на Западных рынках, на Российском всё только начинается.


Поэтому и приняли решение сделать торгового советника, который использует нейросети для прогнозирования изменения цены акций и отображает свои прогнозы на дашборде.

**Цвет ячейки отражает прогноз по этому тикеру, поэтому даже выросшие в цене тикеры - могут упасть)))**

![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/dashboard.jpg)



#### Какие есть скрытые цели? )))
Т.к. этот пример торгового советника с использованием нейросетей хорошо документирован и последовательно проходит через все этапы:
    
- получение исторических данных по акциям
```1_get_historical_data_for_strategy_from_moex.py```
- изучение данных, выбор метрик 
```2_choosing_metrics_for_neural_network.ipynb```
- подготовка датасета и обучение нейросети для одной акции
```3_train_neural_network_for_one_symbol.py```
- проверка предсказаний сделанных нейросетью для одной акции
```4_check_predictions_by_neural_network_for_one_symbol.py```
- подготовка датасета и обучение нейросети для многих акций индекса
```5_train_neural_network_for_many_symbols.py```
- обновление предсказаний нейросетью для акций из индекса в двух режимах работы: live + offline 
```6_live_update_adviser_data.py```
- online визуализация дашборда предсказаний нейросетью для акций из индекса (с автообновлением)
```7_run_adviser_for_investors.py```
- записано обучающее видео как запускать и работать с этим кодом, выложенное [на YouTube](https://youtu.be/yrQFqvc4fk0 ) и [на RuTube](https://rutube.ru/video/private/1255dfe65f4db8736b894cae72b14c45/?p=oOFSDPr1El6lq586tIm2qg )

то, это позволит всем, кто только начинает свой путь по применению нейросетей для аналитики, использовать этот код, 
как стартовый шаблон с последующим его усовершенствованием и допиливанием)) 

- По крайней мере появился +1 рабочий пример использования нейросетей для аналитики цен акций.

Тем самым, станет больше роботов с использованием искусственного интеллекта,
```
- это повлечет большую волатильность нашего фондового рынка
- большую ликвидность за счет большего количества сделок
- и соответственно больший приток капитала в фондовый рынок
```


==========================================================================

## Установка
1) Самый простой способ:
```shell
git clone https://github.com/WISEPLAT/Hackathon-MOEX-NN-Advisor
```

2) Или через PyCharm:
- нажмите на кнопку **Get from VCS** при создании нового проекта.

Вот ссылка на этот проект:
```shell
https://github.com/WISEPLAT/Hackathon-MOEX-NN-Advisor
```

### Установка дополнительных библиотек
Для работы торгового советника с использованием нейросетей, есть некоторые библиотеки, которые вам необходимо установить,
их можно установить такой командой:
```shell
pip install numpy pandas "tensorflow<2.11" backtrader moexalgo flask
```

или так:
```shell
pip install -r requirements.txt
```

Обязательно! Выполните в корне вашего проекта через терминал эту команду:
```shell
git clone https://github.com/WISEPLAT/backtrader_moexalgo
```
для клонирования библиотеки, которая позволяет работать с функционалом API AlgoPack + Backtrader.

Теперь наш проект выглядит вот так:
![alt text](https://raw.githubusercontent.com/WISEPLAT/imgs_for_repos/master/hackathon_moex_nn_advisor.jpg )

Теперь можно запускать и смотреть, а предварительно лучше посмотреть видео по работе с этим кодом
, выложенное [на YouTube](https://youtu.be/yrQFqvc4fk0 ) и [на RuTube](https://rutube.ru/video/private/1255dfe65f4db8736b894cae72b14c45/?p=oOFSDPr1El6lq586tIm2qg )

### Внимание
Некоторые файлы содержат строку:
```exit(777)  # для запрета запуска кода, иначе перепишет результаты```
это сделано специально, чтобы случайно не перезаписать данные, её можно закомментировать, когда будете тестировать свои модели и свои настройки.


Работоспособность проверялась на ```Python 3.9+``` с последними версиями библиотек.


==========================================================================

## Спасибо
- backtrader: очень простая и классная библиотека!
- Команде разработчиков MOEX [moexalgo](https://github.com/moexalgo/moexalgo): Для создания оболочки MOEX API, сокращающей большую часть работы.
- Игорю Чечету: за классные бесплатные библиотеки для live торговли реализующие подключения к брокерам 
- tensorflow: За простую и классную библиотеку для работы с нейросетями.

## Важно
Исправление ошибок, доработка и развитие кода осуществляется автором и сообществом!

**Пушьте ваши коммиты!** 

# Условия использования
Программный код выложенный по адресу https://github.com/WISEPLAT/Hackathon-MOEX-NN-Advisor в сети интернет, реализующий отображение рекомендаций по акциям на фондовом рынке с использованием нейросетей - это **Программа** созданная исключительно для удобства работы и изучения применений нейросетей/искусственного интеллекта.
При использовании **Программы** Пользователь обязан соблюдать положения действующего законодательства Российской Федерации или своей страны.
Использование **Программы** предлагается по принципу «Как есть» («AS IS»). Никаких гарантий, как устных, так и письменных не прилагается и не предусматривается.
Автор и сообщество не дает гарантии, что все ошибки **Программы** были устранены, соответственно автор и сообщество не несет никакой ответственности за
последствия использования **Программы**, включая, но, не ограничиваясь любым ущербом оборудованию, компьютерам, мобильным устройствам, 
программному обеспечению Пользователя вызванным или связанным с использованием **Программы**, а также за любые финансовые потери,
понесенные Пользователем в результате использования **Программы**.
Никто не ответственен за потерю данных, убытки, ущерб, включаю случайный или косвенный, упущенную выгоду, потерю доходов или любые другие потери,
связанные с использованием **Программы**.

**Программа** распространяется на условиях лицензии [MIT](https://choosealicense.com/licenses/mit).

==========================================================================
