from Stock_Visualised import StockAnalysis
from pre_processing import Training, Anamoly

class BaseAPI:
    def __init__(self, data_directory, time_steps, train_test_size, learning_rate, epoch, batch_size, validation_split,
                 loss):
        self.data_directory = data_directory
        self.time_steps = time_steps
        self.train_test_size = train_test_size
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.loss = loss

    def train_base_model(self):
        data = StockAnalysis(self.data_directory)
        dataset = data.read_csv()
        preprocess = Training(dataset, self.train_test_size, self.learning_rate, self.epoch, self.batch_size,
                              self.validation_split, self.loss)
        train, test = preprocess.scaler_transforms()
        X_train, y_train = preprocess.create_dataset(train[['Close']], train.Close, self.time_steps)
        X_test, y_test = preprocess.create_dataset(test[['Close']], test.Close, self.time_steps)
        model, callbacks = preprocess.model(X_train)
        history = preprocess.start_training(model, callbacks, X_train, y_train)
        return model, history


class TestAPI:
    def __init__(self, data_directory, time_steps, train_test_size, learning_rate, epoch, batch_size, validation_split,
                 loss, model):
        self.data_directory = data_directory
        self.time_steps = time_steps
        self.train_test_size = train_test_size
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.loss = loss
        self.model = model
        self.training = Training(self.data_directory, self.train_test_size, self.learning_rate, self.epoch,
                                 self.batch_size, self.validation_split, self.loss)
        self.anamoly = Anamoly(self.data_directory, self.train_test_size, self.learning_rate, self.epoch,
                               self.batch_size,
                               validation_split, loss, self.model)

    def detect_anamoly(self):
        train_scalar, test_scalar = self.anamoly.scalar_transforms()
        x_test_scalar, y_test_scalar = self.training.create_dataset(test_scalar[['Close']], test_scalar.Close,
                                                                    self.time_steps)
        x_train_scalar, y_train_scalar = self.training.create_dataset(train_scalar[['Close']], train_scalar.Close,
                                                                      self.time_steps)
        fit_history = self.anamoly.fit_model(x_train_scalar, y_train_scalar)

        return fit_history



