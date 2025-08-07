import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data_folder = 'trenazniPodaci'
window_size = 10  

all_data = []


for file in os.listdir(data_folder):
    if file.endswith('.xlsx'):
        path = os.path.join(data_folder, file)
        df = pd.read_excel(path)

        print(df[['Date', 'Time']].head())

        
        if not {'Date', 'Time', 'Open', 'Last'}.issubset(df.columns):
            continue

        # df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
        df['Datetime'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Datetime')

        df['log_return'] = np.log(df['Last'] / df['Open'])
        df['volatility'] = df['log_return'].rolling(window=window_size).std()

        df.dropna(inplace=True)

        all_data.append(df[['Open', 'Last', 'volatility']])


combined_df = pd.concat(all_data, ignore_index=True)


X = combined_df[['Open', 'Last']]
y = combined_df['volatility']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1) 
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])


history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Stvarna volatilnost")
plt.ylabel("Predviđena volatilnost")
plt.title("Predikcija volatilnosti neuralnom mrežom")
plt.grid(True)
plt.show()
