from Stock_Visualised import StockAnalysis
from pre_processing import Training, Anamoly

DATA_DIR = "StockData/JNJ.csv"
TEST_DATA_DIR = "StockData/BSE Sensex.csv"
TIME_STEPS = 15
TRAIN_TEST_SIZE = 0.9
LEARNING_RATE = 0.001
EPOCH = 100
BATCH_SIZE = 2
VALIDATION_SPLIT = 0.1
LOSS = 'mae'


def train_base_model(data_directory, time_steps, train_test_size, learning_rate, epoch, batch_size, validation_split,
                     loss):
    data = StockAnalysis(data_directory)
    dataset = data.read_csv()
    preprocess = Training(dataset, train_test_size, learning_rate, epoch, batch_size, validation_split, loss)
    train, test = preprocess.scaler_transforms()
    X_train, y_train = preprocess.create_dataset(train[['Close']], train.Close, time_steps)
    X_test, y_test = preprocess.create_dataset(test[['Close']], test.Close, time_steps)
    model, callbacks = preprocess.model(X_train)
    history = preprocess.start_training(model, callbacks, X_train, y_train)


def detect_anamoly(data_directory, time_steps, train_test_size, learning_rate, epoch, batch_size, validation_split,
                   loss):
    preprocess = Training(data_directory, train_test_size, learning_rate, epoch, batch_size, validation_split, loss)
    anamoly = Anamoly(data_directory, train_test_size, learning_rate, epoch, batch_size, validation_split, loss)
    train_scalar, test_scalar = anamoly.scalar_transforms()
    x_test_scalar, y_test_scalar = preprocess.create_dataset(test_scalar[['Close']], test_scalar.Close, time_steps)
    x_train_scalar, y_train_scalar = preprocess.create_dataset(train_scalar[['Close']], train_scalar.Close,
                                                           time_steps)
    fit_history = anamoly.fit_model(x_train_scalar, y_train_scalar)


# train_base_model(DATA_DIR, TIME_STEPS, TRAIN_TEST_SIZE, LEARNING_RATE, EPOCH, BATCH_SIZE, VALIDATION_SPLIT, LOSS)

detect_anamoly(TEST_DATA_DIR, TIME_STEPS, TRAIN_TEST_SIZE, LEARNING_RATE, EPOCH, BATCH_SIZE, VALIDATION_SPLIT, LOSS)
