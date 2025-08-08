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

jump_threshold = 0.0025  # prag za jump, 2%
hour_limit = 0.05         # ako želiš vremensko ograničenje u satima

# Izračunaj razliku u satima između uzastopnih merenja
df["hour_diff"] = df["Datetime"].diff().dt.total_seconds() / 3600

# Obeleži jump-ove
df["is_jump"] = df["Return"].abs() > jump_threshold


# Pretpostavljamo da je df već kreiran i 'is_jump' kolona postoji

# Izračunaj sat iz BarNo (5-minutni barovi)
df['Hour'] = 9 + (df['BarNo'] - 1) * 5 / 60  # Počinje u 9h

# Filtriraj samo jumpove
jumps_df = df[df["is_jump"]]

# Podeli na pozitivne i negativne skokove
pos_jumps = jumps_df[jumps_df["DeseasonedReturn"] > 0]
neg_jumps = jumps_df[jumps_df["DeseasonedReturn"] < 0]

# Grupisi po satu i izračunaj prosečan jump (u bp)
pos_avg = pos_jumps.groupby('Hour')["DeseasonedReturn"].mean() * 10000
neg_avg = neg_jumps.groupby('Hour')["DeseasonedReturn"].mean() * 10000

# Plot
fig, axs = pyplot.subplots(1, 2, figsize=(12, 5), sharex=True)

# Pozitivni skokovi
axs[0].bar(pos_avg.index, pos_avg.values, color='pink')
axs[0].set_title("Positive jumps", fontsize=12, fontweight='bold')
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Average jump size (bp)")

# Negativni skokovi
axs[1].bar(neg_avg.index, neg_avg.values, color='pink')
axs[1].set_title("Negative jumps", fontsize=12, fontweight='bold')
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Average jump size (bp)")

pyplot.tight_layout()
pyplot.show()

# hours = [9.5, 10, 11, 12, 13, 14, 15, 16]
# pos_avg = [5, 8, 3, 4, 2, 7, 9]
# neg_avg = [-6, -9, -4, -2, -3, -8, -10]

# fig, axs = pyplot.subplots(1, 2, figsize=(12, 5), sharex=True)

# axs[0].bar(hours, pos_avg, color='gray')
# axs[0].set_title("Positive jumps", fontsize=12, fontweight='bold')
# axs[0].set_xlabel("Time")
# axs[0].set_ylabel("Average jump size (bp)")

# axs[1].bar(hours, neg_avg, color='gray')
# axs[1].set_title("Negative jumps", fontsize=12, fontweight='bold')
# axs[1].set_xlabel("Time")
# axs[1].set_ylabel("Average jump size (bp)")

# pyplot.tight_layout()
# pyplot.show()

profit_threshold = 0.1   # +10%
loss_threshold = -0.1    # -10%
exit_bar = 78            # Time stop: pre poslednjeg bara, ima ih 79
df["StraddlePrice"] = df["Last"] * 0.02

trades = []

for day, day_df in df.groupby("date_only"):
    day_df = day_df.sort_values("BarNo").copy().reset_index(drop=True)

    #pretpostavka da se kupuje u prvih 20 bara
    #print(f"\n{day} - broj barova:", len(day_df))

    # jump_signal = day_df["is_jump"]

    
    jump_signal = day_df["is_jump"]

    #print("Broj skokova u prvih 20 barova:", jump_signal.sum())

    if jump_signal.sum() > 0:

        entry_index = jump_signal[jump_signal].index[0]
        entry_price = day_df.loc[entry_index, "StraddlePrice"]
        entry_time = day_df.loc[entry_index, "Datetime"]
        entry_barno = day_df.loc[entry_index, "BarNo"]
        entry_underlying = day_df.loc[entry_index, "Last"]

        
        open_trade = True
        for i in range(entry_index + 1, day_df.index[-1] + 1):
            price_now = day_df.loc[i, "Last"]
            # Model straddle vrednosti: proporcionalno apsolutnoj promeni cene
            price_change = abs(price_now - entry_underlying) / entry_underlying
            straddle_value = entry_price + entry_price * price_change

            pnl = (straddle_value - entry_price) / entry_price

            if pnl >= profit_threshold or pnl <= loss_threshold or day_df.loc[i, "BarNo"] >= exit_bar:
                exit_time = day_df.loc[i, "Datetime"]
                trades.append({
                    "date": day,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry_price": entry_price,
                    "exit_price": straddle_value,
                    "pnl": pnl
                })
                open_trade = False
                break

        # Ako nije zatvorena pozicija do kraja dana
        if open_trade:
            i = day_df.index[-1]
            price_now = day_df.loc[i, "Last"]
            price_change = abs(price_now - entry_underlying) / entry_underlying
            straddle_value = entry_price + entry_price * price_change
            pnl = (straddle_value - entry_price) / entry_price
            trades.append({
                "date": day,
                "entry_time": entry_time,
                "exit_time": day_df.loc[i, "Datetime"],
                "entry_price": entry_price,
                "exit_price": straddle_value,
                "pnl": pnl
            })
trades_df = pd.DataFrame(trades)
print(trades_df)
print("\nUkupan broj trejdova:", len(trades_df))
if not trades_df.empty and "pnl" in trades_df.columns:
    print("Prosečan PnL:", trades_df["pnl"].mean())
else:
    print("Nema nijednog trejda ili kolona 'pnl' ne postoji.")

# print("Prosečan PnL:", trades_df["pnl"].mean())
