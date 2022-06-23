import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

DATA_DIR = "StockData/JNJ.csv"


def create_dataset(directory):
    df = pd.read_csv(directory)
    df = df[['Date', 'Close', 'Open', 'High', 'Low']]
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'].min(), df['Date'].max()
    print(df.head())
    return df


def basic_eda(data_frame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_frame['Date'], y=data_frame['Close'], name='Close price'))
    fig.update_layout(showlegend=True, title='JNJ 1985-2021')
    fig.show()


def candlestick_eda(data_frame):
    fig = go.Figure(data=[go.Candlestick(x=data_frame.index,
                                         open=data_frame['Open'],
                                         high=data_frame['High'],
                                         low=data_frame['Low'],
                                         close=data_frame['Close'])])
    fig.show()


def table_eda(data_frame):
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(['Date', 'Open', 'High', 'Low', 'Close']),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[data_frame.index, data_frame.Open, data_frame.High, data_frame.Low, data_frame.Close],
                   fill_color='lavender',
                   align='left'))
    ])

    fig.show()


data = create_dataset(DATA_DIR)
basic_eda(data)
candlestick_eda(data)
table_eda(data)