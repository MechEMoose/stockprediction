import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import matplotlib.dates as mdates
import math
from mpl_finance import candlestick_ohlc
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


ticker = input('Enter stock ticker symbol: ')

#Define time frame
start = dt.datetime(2018, 1, 1)
end = dt.datetime.now()

#Load data
data = web.DataReader(ticker, 'yahoo', start, end)

# Dataframe
df = data[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close',]]
df['HL_PCT'] = (df['High'] - df['Close']) / df['Close']*100.0
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]

forecast_col = 'Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)


X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]


df.dropna(inplace=True)

y = np.array(df['label'])

#Split test and training data to 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Machine Learning Classifier: Support Vector Machine (SVM)
clf_svm = svm.SVR(kernel='poly')

#Train SVM Classifier
clf_svm.fit(X_train, y_train)
accuracy_svm = clf_svm.score(X_test, y_test)

#Machine Learning Classifier: Linear Regression (LR)
clf_lr = LinearRegression()

#Train LR Classifier
clf_lr.fit(X_train, y_train)
accuracy_lr = clf_lr.score(X_test, y_test)

forecast_set_svm = clf_svm.predict(X_lately)
print(forecast_set_svm, accuracy_svm, forecast_out)
df['SVM Forecast'] = np.nan

forecast_set_lr = clf_lr.predict(X_lately)
print(forecast_set_lr, accuracy_lr, forecast_out)
df['LR Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set_svm:
    next_date = dt.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

for i in forecast_set_lr:
    next_date = dt.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


#Prepare Data for LSTM
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = int(input('Enter number of prediction days: '))

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build the model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) # Prediction of the next closing cost

model.compile(optimizer='adam', loss='mean_squared_error')
eps = int(input('Enter number of epochs for LSTM model: '))
model.fit(x_train, y_train, epochs=eps, batch_size=32)

''' Test the model accuracy on existing data '''

#Load test data
test_start = start
test_end = end

test_data = web.DataReader(ticker, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

#Make predictions on test data

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# RSI Calculation

delta = data['Adj Close'].diff(1)
delta.dropna(inplace=True)

positive = delta.copy()
negative = delta.copy()

positive[positive < 0] = 0
negative[negative > 0] = 0

days = input('Enter number of days for RSI Calc: ')
days = int(days)

average_gain = positive.rolling(window=days).mean()
average_loss = abs(negative.rolling(window=days).mean())
relative_strength = average_gain / average_loss

RSI = 100.0 - (100.0 / (1.0 + relative_strength))

combined = pd.DataFrame()
combined ['Adj Close'] = data['Adj Close']
combined['RSI'] = RSI

#Restructure Data
data = data[['Open', 'High', 'Low', 'Close']]
data.reset_index(inplace=True)
data['Date'] = data['Date'].map(mdates.date2num)

#Predict next day
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)

print('Last Close: $', data['Close'].iloc[-1])
print('Linear Regression Prediction: ', forecast_set_lr[0])
print('Linear Regression Prediction Confidence: ', accuracy_lr*100,'%')
print('SVM Prediction: ', forecast_set_svm[0])
print('SVM Prediction Confidence: ', accuracy_svm*100, '%')
print(f"LSTM Prediction: {prediction}")


#Plot of graphs
fig, (ax1, ax2) = plt.subplots(2)


ax1.plot(combined.index, combined['Adj Close'], color='lightgray')
ax1.set_title('{} Share Price'.format(ticker), color='white')
ax1.grid(True, color='#555555')
ax1.set_axisbelow(True)
ax1.set_facecolor('black')
ax1.figure.set_facecolor('#121212')
ax1.tick_params(axis = 'x', colors = 'white')
ax1.tick_params(axis = 'y', colors = 'white')
ax1.xaxis_date()
candlestick_ohlc(ax1, data.values, width=0.5, colorup='#00ff00')
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
# ax1.text(0.05, 0.95, ('Tomorrows predicted stock price via LSTM: ', prediction), transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

ax2.plot(combined.index, combined['RSI'], color='lightgray')
ax2.set_title("RSI Value", color='white')
ax2.grid(False)
ax2.set_axisbelow(True)
ax2.set_facecolor('black')
ax2.tick_params(axis = 'x', colors = 'white')
ax2.tick_params(axis = 'y', colors = 'white')
ax2.axhline(0, linestyle='--', alpha=0.5, color='#ff0000')
ax2.axhline(10, linestyle='--', alpha=0.5, color='#ffaa00')
ax2.axhline(20, linestyle='--', alpha=0.5, color='#00ff00')
ax2.axhline(30, linestyle='--', alpha=0.5, color='#cccccc')
ax2.axhline(70, linestyle='--', alpha=0.5, color='#cccccc')
ax2.axhline(80, linestyle='--', alpha=0.5, color='#00ff00')
ax2.axhline(90, linestyle='--', alpha=0.5, color='#ffaa00')
ax2.axhline(100, linestyle='--', alpha=0.5, color='#ff0000')



plt.show()

