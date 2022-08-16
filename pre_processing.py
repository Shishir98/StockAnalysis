from Stock_Visualised import StockAnalysis
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow import keras

TRAIN_TEST_SIZE = 0.9
LEARNING_RATE = 0.001
EPOCH = 100
BATCH_SIZE = 2
VALIDATION_SPLIT = 0.1
LOSS = 'mae'
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
        model.save("Models/version1_test")
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend();
        plt.show()
        return history,model
