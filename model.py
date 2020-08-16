import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import plotly.graph_objects as go
import plotly.offline as py
import matplotlib.pyplot as plt
import math

import pickle

class Helper:
    def mean_absolute_percentage_error(self,y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def generateDataSet(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def train_LSTM_model(self,X,y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = False, stratify = None,random_state=42)

        X_train = X_train.values.reshape((X_train.values.shape[0], 1, X_train.shape[1]))
        X_test = X_test.values.reshape((X_test.values.shape[0], 1, X_test.shape[1]))

        # design network
        model = keras.Sequential()
        model.add(layers.LSTM(50, activation='linear',input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(layers.Dense(50,activation='linear'))
        model.add(layers.Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        # fit network
        history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=0, shuffle=False)
        # plot history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        predictions = model.predict(X_test)

        actual = y_test

        # calculate RMSE
        rmse = math.sqrt(mean_squared_error(actual, predictions))
        print('Test RMSE: %.3f' % rmse)

        pct = self.mean_absolute_percentage_error(actual,predictions)
        print('MAPE : %.3f' % pct +'%' )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x = np.arange(len(y_train)),
                                     y = y_train,
                                     mode='lines+markers', name='Train'))
        fig.add_trace(go.Scatter(x = np.arange(len(X_train),len(X_train)+len(actual)),
                                     y = actual,
                                     mode='lines+markers', name='Actual'))
        fig.add_trace(go.Scatter(x = np.arange(len(X_train),len(X_train)+len(actual)),
                                     y = predictions,
                                     mode='lines+markers', name='Prediction'))
        fig.update_layout(title="Mean Abs. Pct Error " + '%'+str(round(pct,2)),
                            xaxis_title="Months",
                            yaxis_title="Amount")
        fig.show()

        return model


def main():
    usd=pd.read_csv('usdtry.csv')

    dataset = pd.DataFrame()

    dataset['USD'] = usd.Price

    helper = Helper()
    reframed = helper.generateDataSet(dataset, 2, 1)

    y = pd.DataFrame(reframed['var1(t)'])
    X = reframed.drop(columns=['var1(t)'])

    model = helper.train_LSTM_model(X,y)
    model.save("model")
    #pickle.dump(model,open('model.pkl','wb'))

if __name__ == '__main__':
    main()
