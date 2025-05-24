import pandas as pd
import numpy as np

df = pd.read_csv('initial_data.csv')

#dropping columns
df = df.drop("HWY (L/100 km)", axis='columns')
df = df.drop("COMB (L/100 km)", axis='columns')
df = df.drop("COMB (mpg)", axis='columns')

print(df.head())

#simulating missing values
df.loc[df.sample(frac = 0.1).index, 'FUEL'] = np.nan
df.loc[df.sample(frac = 0.02).index, 'EMISSIONS'] = np.nan

#repairing back missing values
average_emissions = int(df['EMISSIONS'].mean())
df['EMISSIONS'].fillna(average_emissions, inplace=True)
df['FUEL'].fillna(df['FUEL'].mode()[0], inplace=True)

#removing data outliers
df = df[(df['CYLINDERS'] >= 2) & (df['CYLINDERS'] <= 12)]
df = df[(df['ENGINE SIZE'] >= 0.8) & (df['ENGINE SIZE'] <= 10)]
df = df[(df['FUEL CONSUMPTION'] >= 1) & (df['FUEL CONSUMPTION'] <= 50)]
df = df[df['EMISSIONS'] >= 0]
df = df[(df['FUEL']  == 'X' ) | (df['FUEL']  == 'Z' ) | (df['FUEL']  == 'D' ) | (df['FUEL']  == 'E' ) | (df['FUEL']  == 'N' )]

#adding IMPACT column
avrg = df['EMISSIONS'].mean() * df['ENGINE_SIZE'].mean()
impact_score = df['EMISSIONS'] * df['ENGINE_SIZE']
df['IMPACT'] = ['HIGH' if x else 'LOW' for x in impact_score > avrg]

