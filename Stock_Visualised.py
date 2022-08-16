import pandas as pd
import plotly.graph_objects as go


class StockAnalysis:
    def __init__(self, data_dir):
        self.DATA_DIR = data_dir

    def read_csv(self):
        df = pd.read_csv(self.DATA_DIR)
        df = df[['Date', 'Close']]
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'].min(), df['Date'].max()
        return df

    def basic_eda(self, data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close price'))
        fig.update_layout(showlegend=True, title='JNJ 1985-2021')
        fig.show()

    def candlestick_eda(self, data):
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'])])
        fig.show()

    def table_eda(self, data):
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(['Date', 'Open', 'High', 'Low', 'Close']),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[data.index, data.Open, data.High, data.Low, data.Close],
                       fill_color='lavender',
                       align='left'))
        ])

        fig.show()
