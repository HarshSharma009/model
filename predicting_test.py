import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

time_step=100
test = pd.DataFrame(pd.read_csv("test_AAPL.csv"))
train = pd.DataFrame(pd.read_csv("AAPL.csv"))

test.shape

plt.figure(figsize = (12, 8))
plt.subplot(1,1,1)
plt.plot(test.Open.values, color = "red", label = "Open Stock Price")
plt.grid("both")
plt.title("Real Amazon Prices for the next 21 days")
plt.legend()


test.drop(["Volume","Date", "Adj Close", "High", "Low", "Close"], axis = 1, inplace = True)
train.drop(["Volume","Date", "Adj Close", "High", "Low", "Close"], axis = 1, inplace = True)

real_prices = test.values
dataset_total = pd.concat((train["Open"], test["Open"]), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test) - time_step : ].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.fit_transform(inputs)
inputs.shape


x_test = []
for i in range(time_step, inputs.shape[0]):
  x_test.append(inputs[i - time_step : i , 0])
x_test = np.array(x_test)


x_test.shape
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
plt.figure(figsize= (12, 8))
plt.subplot(1,1,1)
plt.plot(real_prices, color = "red", label = "Real Amazon prices")
plt.plot(predicted_prices, color = "blue", label = "Predicted Amazon prices")
plt.title("Amazon Open Stock Prices")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.grid("both")
plt.show()