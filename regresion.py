import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Učitaj podatke
df = pd.read_excel(r"C:\Users\Nevena Perišić\Desktop\ProjekatCompFin\podaci\SPXintraday.xlsx")

df['Datetime'] = pd.to_datetime(df['Date'])

df = df.sort_values('Datetime')

# Napravi BarNo: redni broj bara u svakom danu (5-minutni intervali)
df['date_only'] = df['Datetime'].dt.date
df['BarNo'] = df.groupby('date_only').cumcount() + 1  # počinje od 1 svakog dana

# Izbaci prvi bar svakog dana i barove posle 79
df = df[df['BarNo'] > 1]
df = df[df['BarNo'] <= 79]

# Izračunaj prinos
df['Return'] = df['Last'].pct_change()
df.loc[df['date_only'] != df['date_only'].shift(1), 'Return'] = np.nan  # prvi u danu NaN
df = df.dropna(subset=['Return'])


# Desezoniranje: oduzmi prosečan prinos
average_return = df['Return'].mean()
df['DeseasonedReturn'] = df['Return'] - average_return

# Drugi graf: deseasonirani prinosi po BarNo
pyplot.figure(figsize=(10,4))
pyplot.plot(df['BarNo'], df['DeseasonedReturn'], 'o', label='Desezonirani prinosi')
pyplot.xlabel('BarNo')
pyplot.ylabel('Deseasoned Return')
pyplot.title('Desezonirani prinosi po BarNo')
pyplot.grid(True)
pyplot.legend()
pyplot.show()

# Kvadrat log-desezoniranih prinosa i polinomska regresija 2. stepena
x = df['BarNo'].to_numpy()
y = np.log(np.square(df['DeseasonedReturn'].to_numpy()))

poly = PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(x.reshape(-1, 1))

poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, y)

# Predikcija i prikaz
y_predicted = np.sqrt(np.exp(poly_reg_model.predict(poly_features)))

# Realna volatilnost (apsolutna vrednost deseasoniranog prinosa)
real_vol = np.abs(df['DeseasonedReturn'].to_numpy())
expected_vol = y_predicted

pyplot.figure(figsize=(8,6))
pyplot.scatter(real_vol, expected_vol, alpha=0.6)
pyplot.xlabel('Realna volatilnost')
pyplot.ylabel('Očekivana volatilnost (regresija)')
pyplot.title('Očekivana vs. realna volatilnost (scatter)')
pyplot.grid(True)
pyplot.show()

pyplot.figure(figsize=(10,4))
pyplot.plot(x, y_predicted, 'o', label='Polinomska regresija (2. stepen)')
pyplot.xlabel('Vreme')
pyplot.ylabel('Volatilnost')
pyplot.title('Desezonirana volatilnost')
pyplot.grid(True)
pyplot.legend()
pyplot.show()

# Izračunaj grešku
error = (df['DeseasonedReturn'] - y_predicted)
df['Error'] = error

pyplot.figure(figsize=(10,4))
pyplot.plot(range(len(x)), error, label='Greška')
pyplot.xlabel('Index')
pyplot.ylabel('Greška')
pyplot.title('Greška deseasoniranih prinosa')
pyplot.grid(True)
pyplot.legend()
pyplot.show()