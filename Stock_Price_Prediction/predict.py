import pandas
import yfinance
import datetime
import numpy
from datetime import date, timedelta
import plotly.graph_objects as graph
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM

class Stocks:
    def __init__(self, code, date: tuple):
        try:
            self.company_code = code
            self.data = yfinance.download(
                code,
                start=date[0],
                end=date[1],
                progress=False
            )
            # reformatting data object
            self.data['Date'] = self.data.index
            self.data = self.data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
            self.data.reset_index(drop=True, inplace=True)
            #print(self.data.tail())

        except Exception as error:
            print(error)
            exit(1)

    def plot_graph(self):
        figure = graph.Figure(data=[graph.Candlestick(
            x=self.data['Date'],
            open=self.data['Open'],
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close']
        )])
        figure.update_layout(title=(self.company_code + ' Stock Price Analysis'), xaxis_rangeslider_visible=False)
        figure.show()

    def get_model(self):
        x = self.data[['Open', 'High', 'Low', 'Volume']].to_numpy()
        y = self.data['Close'].to_numpy().reshape(-1, 1)
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        #model.summary()

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(xtrain, ytrain, batch_size=1, epochs=30)

        return model


if __name__ == '__main__':
    today = date.today()
    d2 = today.strftime('%Y-%m-%d')
    d1 = (today - timedelta(days=3000)).strftime('%Y-%m-%d')

    obj = Stocks('NFLX', (d1, d2))
    model = obj.get_model()
    #features = [['Open', 'High', 'Low', 'Volume']]
    features = numpy.array([[585.93, 590.02, 563.20, 15243800]])
    result = model.predict(features)
    print(result)
