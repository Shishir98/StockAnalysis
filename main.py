
from Stock_Visualised import StockAnalysis

DATA_DIR = "StockData/JNJ.csv"

data = StockAnalysis(DATA_DIR)
data.basic_eda()
data.candlestick_eda()
data.table_eda()
