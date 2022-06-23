import pandas as pd
import plotly.graph_objects as go


class StockAnalysis:
    def __init__(self, data_dir):
        self.DATA_DIR = data_dir
        self.df = pd.read_csv(self.DATA_DIR)
        self.df = self.df[['Date', 'Close', 'Open', 'High', 'Low']]
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Date'].min(), self.df['Date'].max()

    def basic_eda(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df['Date'], y=self.df['Close'], name='Close price'))
        fig.update_layout(showlegend=True, title='JNJ 1985-2021')
        fig.show()

    def candlestick_eda(self):
        fig = go.Figure(data=[go.Candlestick(x=self.df.index,
                                             open=self.df['Open'],
                                             high=self.df['High'],
                                             low=self.df['Low'],
                                             close=self.df['Close'])])
        fig.show()

    def table_eda(self):
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(['Date', 'Open', 'High', 'Low', 'Close']),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[self.df.index, self.df.Open, self.df.High, self.df.Low, self.df.Close],
                       fill_color='lavender',
                       align='left'))
        ])

        fig.show()
