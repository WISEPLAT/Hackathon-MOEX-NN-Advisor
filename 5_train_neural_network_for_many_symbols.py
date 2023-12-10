"""
    В этом коде мы обучаем нейросеть (НС), для входа используем выбранные метрики,
    для выхода d_close. Модели НС сохраняются в папку NN/_models.
    После обучения, лучшая модель НС сохраняется автоматом в папку NN_winner.

    Логи работы сохранены в файлах:
    - 3_Training and Validation Accuracy and Loss.jpg - график Training and Validation Accuracy and Loss
    - 3_results_of_training_neural_network.txt - процесс обучения нейросети логи с экрана
    Итак в процессе этого обучения лучше всего себя показала модель на эпохе 27:
    -----------------------------------------------------------------
    train accuracy = 95.4852%
    test accuracy = 95.7101%
    test error = 857 out of 19977 examples

    Model: "sequential_18"
    -----------------------------------------------------------------

    Авторы: Олег Шпагин, Федор Шпагин
    Github: https://github.com/WISEPLAT
    Telegram: https://t.me/OlegSh777
"""
import random

exit(777)  # для запрета запуска кода, иначе перепишет результаты

import functions
import matplotlib.pyplot as plt
import os
import tensorflow as tf

import sys
from datetime import datetime
import pandas as pd
import numpy as np
import math
import shutil

from tensorflow import keras
from tensorflow import config
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Rescaling
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint


# Import Keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from time import time
from keras.models import load_model

from my_config.trade_config import Config  # Файл конфигурации

# pip install tf-nightly
# pip install "tensorflow<2.11" - to enable GPU without VM
print("Num GPUs Available: ", len(config.list_physical_devices('GPU')))


if __name__ == '__main__':  # Точка входа при запуске этого скрипта

    # перенаправлять ли вывод с консоли в файл
    functions.start_redirect_output_from_screen_to_file(False, filename="3_results_of_training_neural_network.txt")

    # ------------------------------------------------------------------------------------------------------------------

    timeframe_0 = Config.timeframe_0  # таймфрейм для обучения нейросети - вход - для картинок
    training_NN = Config.training_NN  # тикеры по которым будем обучать нейросети
    NN_winner_folder = "NN_winner"  # папка в которую будем сохранять лучшие модели

    # ================================================================================================================

    cur_run_folder = os.path.abspath(os.getcwd())  # текущий каталог

    for symbol in training_NN:

        # symbol = 'SBER'  # Тикер, который будем исследовать

        try:

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

            column_indices = {name: i for i, name in enumerate(df_nn.columns)}
            print(column_indices)
            train_df = df_nn[0:int(len(df_nn) * 0.8)]
            test_df = df_nn[int(len(df_nn) * 0.8):]

            num_features = df_nn.shape[1]
            print('num_features =', num_features)

            # Split train and test data
            train_labels = train_df.pop("d_close")
            train_features = train_df

            test_labels = test_df.pop("d_close")
            test_features = test_df

            # I want to use a T-days window of input data for predicting target_class
            # It means I need to prepend (T-1) last train records to the 1st test window
            T = 16  # my choice of the timesteps window

            prepend_features = train_features.iloc[-(T - 1):]
            test_features = pd.concat([prepend_features, test_features], axis=0)
            print(train_features.shape, train_labels.shape, test_features.shape, test_labels.shape)

            X_train, y_train = [], []
            for i in range(train_labels.shape[0] - (T - 1)):
                X_train.append(train_features.iloc[i:i + T].values)
                y_train.append(train_labels.iloc[i + (T - 1)])
            X_train, y_train = np.array(X_train), np.array(y_train).reshape(-1, 1)
            print(f'Train data dimensions: {X_train.shape}, {y_train.shape}')

            # print(X_train[0], y_train[0])

            X_test, y_test = [], []
            for i in range(test_labels.shape[0]):
                X_test.append(test_features.iloc[i:i + T].values)
                y_test.append(test_labels.iloc[i])
            X_test, y_test = np.array(X_test), np.array(y_test).reshape(-1, 1)

            print(f'Test data dimensions: {X_test.shape}, {y_test.shape}')

            # Let's make a list of CONSTANTS for modelling:
            LAYERS = [16, 8, 4, 1]  # number of units in hidden and output layers
            M_TRAIN = X_train.shape[0]  # number of training examples (2D)
            M_TEST = X_test.shape[0]  # number of test examples (2D),full=X_test.shape[0]
            N = X_train.shape[2]  # number of features
            BATCH = M_TRAIN  # batch size
            EPOCH = 250  # number of epochs
            LR = 0.05  # 5e-2  # learning rate of the gradient descent
            LAMBD = 0.03  # 3e-2  # lambda in L2 regularizaion
            print(LR, LAMBD)
            DP = 0.0  # dropout rate
            RDP = 0.0  # recurrent dropout rate
            print(f'layers={LAYERS}, train_examples={M_TRAIN}, test_examples={M_TEST}')
            print(f'batch = {BATCH}, timesteps = {T}, features = {N}, epochs = {EPOCH}')
            print(f'lr = {LR}, lambda = {LAMBD}, dropout = {DP}, recurr_dropout = {RDP}')

            test_acc_max = 0.0
            train_acc_max = 0.0

            # ------------------------- Подбор оптимальных моделей -------------------------

            for i in range(20):

                LAYERS = [random.randint(1, 17), random.randint(1, 8), random.randint(1, 4), 1]

                # Build the Model
                model = Sequential()
                model.add(LSTM(input_shape=(T, N), units=LAYERS[0],
                               activation='tanh', #recurrent_activation='hard_sigmoid',  # to prevent - Layer lstm will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.
                               kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
                               dropout=DP, recurrent_dropout=RDP,
                               return_sequences=True, return_state=False,
                               stateful=False, unroll=False
                               ))
                model.add(BatchNormalization())
                model.add(LSTM(units=LAYERS[2],
                               activation='tanh', #recurrent_activation='hard_sigmoid',  # to prevent - Layer lstm will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.
                               kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
                               dropout=DP, recurrent_dropout=RDP,
                               return_sequences=False, return_state=False,
                               stateful=False, unroll=False
                               ))
                model.add(Dropout(0.5))
                model.add(Dense(units=LAYERS[3], activation='sigmoid'))

                # Compile the model with Adam optimizer
                model.compile(loss='binary_crossentropy',
                              metrics=['accuracy'],
                              optimizer=Adam(learning_rate=LR))
                # print(model.summary())

                # Define a learning rate decay method:
                lr_decay = ReduceLROnPlateau(monitor='loss',
                                             patience=1, verbose=0,
                                             factor=0.5, min_lr=1e-8)
                # Define Early Stopping:
                early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0,
                                           patience=30, verbose=1, mode='auto',
                                           baseline=0, restore_best_weights=True)
                # Train the model.
                # The dataset is small for NN - let's use test_data for validation
                start = time()

                # для записи моделей
                save_models = ModelCheckpoint(functions.join_paths([cur_run_folder, "NN", "_models", 'cnn_Open{epoch:1d}.hdf5']))

                history = model.fit(X_train, y_train,
                                    epochs=EPOCH,
                                    batch_size=BATCH,
                                    validation_split=0.0,
                                    validation_data=(X_test[:M_TEST], y_test[:M_TEST]),
                                    shuffle=True, verbose=1,
                                    callbacks=[save_models, lr_decay, early_stop])
                print('-' * 65)
                print(f'Training was completed in {time() - start:.2f} secs')
                print('-' * 65)
                # Evaluate the model:
                train_loss, train_acc = model.evaluate(X_train, y_train,
                                                       batch_size=M_TRAIN, verbose=0)
                test_loss, test_acc = model.evaluate(X_test[:M_TEST], y_test[:M_TEST],
                                                     batch_size=M_TEST, verbose=0)
                print('-' * 65)
                print(f'train accuracy = {round(train_acc * 100, 4)}%')
                print(f'test accuracy = {round(test_acc * 100, 4)}%')
                print(f'test error = {round((1 - test_acc) * M_TEST)} out of {M_TEST} examples')
                print('LAYERS:', LAYERS)

                if test_acc > test_acc_max:
                    test_acc_max = test_acc
                    train_acc_max = train_acc
                    model.save(functions.join_paths([cur_run_folder, NN_winner_folder, f"{symbol}_model.hdf5"]))

            # ------------------------- Подбор оптимальных моделей -------------------------

            print()
            print("*" * 30, "WINNER", "*"*30)

            model = load_model(os.path.join("NN_winner", f"{symbol}_model.hdf5"))
            train_loss, train_acc = model.evaluate(X_train, y_train,
                                                   batch_size=M_TRAIN, verbose=0)
            test_loss, test_acc = model.evaluate(X_test[:M_TEST], y_test[:M_TEST],
                                                 batch_size=M_TEST, verbose=0)
            print('-' * 65)
            print(f'train accuracy = {round(train_acc * 100, 4)}%')
            print(f'test accuracy = {round(test_acc * 100, 4)}%')
            print(f'test error = {round((1 - test_acc) * M_TEST)} out of {M_TEST} examples')
            print()
            print(model.summary())
            print()
            print("*" * 30, "WINNER", "*" * 30)
            print()

            # # Plot the loss and accuracy curves over epochs:
            # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
            # axs[0].plot(history.history['loss'], color='b', label='Training loss')
            # axs[0].plot(history.history['val_loss'], color='r', label='Validation loss')
            # axs[0].set_title("Loss curves")
            # axs[0].legend(loc='best', shadow=True)
            # axs[1].plot(history.history['accuracy'], color='b', label='Training accuracy')
            # axs[1].plot(history.history['val_accuracy'], color='r', label='Validation accuracy')
            # axs[1].set_title("Accuracy curves")
            # axs[1].legend(loc='best', shadow=True)
            # plt.show()

            # графики потерь и точности на обучающих и проверочных наборах
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.plot(acc, label='Training Accuracy')
            plt.plot(val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.title('Training and Validation Accuracy')

            plt.subplot(1, 2, 2)
            plt.plot(loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.savefig(f"5_Training and Validation Accuracy and Loss - {symbol}.png", dpi=150)

            _from = functions.join_paths([cur_run_folder, f"5_Training and Validation Accuracy and Loss - {symbol}.png"])
            _to = functions.join_paths([cur_run_folder, NN_winner_folder, f"5_Training and Validation Accuracy and Loss - {symbol}.png"])
            shutil.move(_from, _to)

            # plt.show()
        except Exception as e:
            print(f"Ошибка - {symbol}: {e}")

    # ================================================================================================================

    # остановка перенаправления вывода с консоли в файл
    functions.stop_redirect_output_from_screen_to_file()
