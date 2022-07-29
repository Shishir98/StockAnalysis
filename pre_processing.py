from Stock_Visualised import StockAnalysis
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow import keras


class Training():
    def __init__(self, data):
        self.data = data

    def scaler_transforms(self):
        train_size = int(len(self.data) * 0.9)
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
        opt = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss='mae')
        callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
        model.summary()
        return model, callbacks

    def start_training(self, model, callbacks, x_train, y_train):
        history = model.fit(
            x_train, y_train,
            epochs=100,
            batch_size=2,
            validation_split=0.1,
            callbacks=[callbacks],
            shuffle=False
        )
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend();
        plt.show()
        return history
