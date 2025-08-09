import os
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data_folder = 'trenazniPodaci'
window_size = 10  

all_data = []
all_data_time = []

for file in os.listdir(data_folder):
    if file.endswith('.xlsx'):
        path = os.path.join(data_folder, file)
        df = pd.read_excel(path)

        #print(df[['Date', 'Time']].head())

        if not {'Date', 'Time', 'Open', 'Last'}.issubset(df.columns):
            continue

        # df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
        df['Datetime'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Datetime')

        
        #TODO: fale kolone
        # print("##########################################")
        # print(df['Date'].size)
        # print(df['Date'])
        # print("##########################################")

        #df['log_return'] = np.log(df['Last'] / df['Open'])
        df['log_return'] = np.log(df['Last'] / df['Last'].shift(1))
        df['volatility'] = df['log_return'].rolling(window=window_size).std()

        df.dropna(inplace=True)

        all_data.append(df[['Open', 'Last', 'volatility']])
        all_data_time.append(df[['Datetime', 'volatility']])
        #print(df['volatility'])


combined_df = pd.concat(all_data, ignore_index=True)

combined_df_time = pd.concat(all_data_time, ignore_index=True)
# combined_df_time['hour'] = combined_df_time['Datetime'].dt.floor('5T')

# ...nakon print("MSE:", mse)

# Grupisanje po 5-minutnim intervalima u danu (bez obzira na datum)
combined_df_time['time_5min'] = combined_df_time['Datetime'].dt.strftime('%H:%M')
avg_vol_per_5min = combined_df_time.groupby('time_5min')['volatility'].mean()


# Priprema x ose sa fiksnim datumom 2024-01-02
x_ticks = pd.date_range('2024-01-02 09:30', '2024-01-02 16:00', freq='5T')
x_labels = [dt.strftime('%H:%M') for dt in x_ticks]
y_vals = [avg_vol_per_5min.get(label, np.nan) for label in x_labels]

plt.figure(figsize=(14, 6))
plt.plot(x_ticks, y_vals, marker='o', linestyle='-')
plt.xlabel("Vreme (5-minutni intervali, 2024-01-02)")
plt.ylabel("Prosečna volatilnost")
plt.title("Prosečna volatilnost po 5-minutnim intervalima (svi dani)")
plt.grid(True)
plt.xticks(x_ticks[::3], x_labels[::3], rotation=45)  # ređe labele radi preglednosti
plt.tight_layout()
#plt.show()

X = combined_df[['Open', 'Last']]
y = combined_df['volatility']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='softplus') 
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])


history = model.fit(X_train, y_train, epochs=160, validation_split=0.2, verbose=1)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)


plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Stvarna volatilnost")
plt.ylabel("Predviđena volatilnost")
plt.title("Predikcija volatilnosti neuralnom mrežom")
plt.grid(True)
plt.show()

# x_ticks = pd.date_range('2024-01-02 09:30', '2024-01-02 16:00', freq='5T')
# x_labels = [dt.strftime('%H:%M') for dt in x_ticks]
# y_vals = [avg_vol_per_hour.get(label, np.nan) for label in x_labels]



# plt.figure(figsize=(10, 5))
# plt.plot(avg_vol_per_hour.index, avg_vol_per_hour.values, marker='o')
# plt.xlabel("Sat u danu")
# plt.ylabel("Prosečna volatilnost")
# plt.title("Prosečna volatilnost po satima (svi dani)")
# plt.grid(True)
# plt.xticks(x_ticks[::3], x_labels[::3], rotation=45)  # ređe labele radi preglednosti
# plt.tight_layout()
# plt.show()