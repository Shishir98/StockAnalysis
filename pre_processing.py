import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential


class Training():
    def __init__(self, data, train_test_size, learning_rate, epoch, batch_size, validation_split, loss):
        self.data = data
        self.train_test_size = train_test_size
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.loss = loss

    def scaler_transforms(self):
        train_size = int(len(self.data) * self.train_test_size)
        test_size = len(self.data) - train_size
        train, test = self.data.iloc[0:train_size], self.data.iloc[train_size:len(self.data)]
        print(train.shape, test.shape)
        scaler = StandardScaler()
        scaler = scaler.fit(train[['Close']])

        train['Close'] = scaler.transform(train[['Close']])
        test['Close'] = scaler.transform(test[['Close']])

        return train, test

    def create_dataset(self, X, y, timestep=1):
        Xs, ys = [], []
        for i in range(len(X) - timestep):
            v = X.iloc[i:(i + timestep)].values
            Xs.append(v)
            ys.append(y.iloc[i + timestep])
        return np.array(Xs), np.array(ys)

    def model(self, train_data):
        model = Sequential()
        model.add(LSTM(64, input_shape=(train_data.shape[1], train_data.shape[2])))
        model.add(Dropout(rate=0.2))
        model.add(RepeatVector(train_data.shape[1]))
        # model.add(LSTM(128, return_sequences=True))
        # model.add(Dropout(rate=0.2))
        model.add(TimeDistributed(Dense(train_data.shape[2])))
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss=self.loss)
        callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
        model.summary()
        return model, callbacks

    def start_training(self, model, callbacks, x_train, y_train):
        history = model.fit(
            x_train, y_train,
            epochs=self.epoch,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[callbacks],
            shuffle=False
        )
        model.save("Models/version1")
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend();
        plt.show()
        return history, model


class Anamoly():
    def __init__(self, data, train_test_size, learning_rate, epoch, batch_size, validation_split, loss,model):
        self.model = model
        self.data = data
        self.train_test_size = train_test_size
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.loss = loss

    def scalar_transforms(self):
        scaler = StandardScaler()
        test_df_final = pd.read_csv(self.data, parse_dates=['Date'])
        train_size_final = int(len(test_df_final) * 0.60)
        test_size_final = len(test_df_final) - train_size_final
        train_df_final, test_df_final = test_df_final.iloc[0:train_size_final], test_df_final.iloc[
                                                                                train_size_final:len(test_df_final)]
        scaler = scaler.fit(train_df_final[['Close']])
        train_df_final['Close'] = scaler.transform(train_df_final[['Close']])
        test_df_final['Close'] = scaler.transform(test_df_final[['Close']])
        print(type(train_df_final), type(test_df_final))

        return train_df_final, test_df_final

    def create_test_dataset(self, train_df_final, test_df_final, timestep=1):
        X_test_final, y_test_final = Training.create_dataset(test_df_final[['Close']], test_df_final.Close, timestep)
        X_train_final, y_train_final = Training.create_dataset(train_df_final[['Close']], train_df_final.Close,
                                                               timestep)
        return X_test_final, y_test_final, X_train_final, y_train_final

    def fit_model(self, X_train_final, y_train_final):
        # model = keras.models.load_model('Models/version1_test')
        callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
        history_final = self.model.fit(
            X_train_final, y_train_final,
            epochs=100,
            batch_size=2,
            validation_split=0.1,
            callbacks=[callbacks],
            shuffle=False
        )
        return history_final
