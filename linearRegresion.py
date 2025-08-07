import pandas as pd
import os
import datetime
import numpy as np
import matplotlib.pyplot as pyplot

#load data
df = pd.read_excel('C:\Users\Nevena Perišić\Desktop\ProjekatCompFin\podaci\SPXintraday.xlsx')

#calculate returns
df['Return'] = df['Last'].pct_change()
df = df.drop(df[df['BarNo'] == 1.0].index)    
df = df.drop(df[df['BarNo'] > 79].index)

# plot return as y and BarNo as x
pyplot.plot(df['BarNo'], df['Return'], 'o')

average_return = df['Return'].mean()
df['Return'] = df['Return'] - average_return

# plot deseasoned return as y and BarNo as x
pyplot.plot(df['BarNo'], df['Return'], 'o')