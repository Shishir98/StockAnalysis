from Stock_Visualised import StockAnalysis
from pre_processing import Training

DATA_DIR = "StockData/JNJ.csv"
time_steps = 15

data = StockAnalysis(DATA_DIR)


dataset = data.read_csv()

preprocess = Training(dataset)

train, test = preprocess.scaler_transforms()

X_train, y_train = preprocess.create_dataset(train[['Close']], train.Close, time_steps)
X_test, y_test = preprocess.create_dataset(test[['Close']], test.Close, time_steps)

model, callbacks = preprocess.model(X_train)

history = preprocess.start_training(model, callbacks, X_train, y_train)