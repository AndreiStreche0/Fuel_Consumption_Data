import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def drop_columns(df):
    df = df.drop("HWY (L/100 km)", axis='columns')
    df = df.drop("COMB (L/100 km)", axis='columns')
    df = df.drop("COMB (mpg)", axis='columns')
    return df

def repair_data(df):
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
    
    return df

def print_stats(df, initial_row_number, empty_count):
    #number of missing values 
    final_row_number = len(df)
    missing_number = initial_row_number - final_row_number
    percentile_missing = missing_number / initial_row_number * 100

    print(f"The number of empty values was: {empty_count}")
    print(f"The number of removed values is {missing_number}, {percentile_missing} of total")

    #getting statistics

    print("Descriptive statistics")
    print(df.describe())
    print(df.describe(include='object'))

def print_distribution(df):
    #histogram
    hist_columns = ['YEAR', 'ENGINE SIZE', 'CYLINDERS', 'FUEL CONSUMPTION', 'EMISSIONS']
    df[hist_columns].hist(bins = 20)
    
    #countplot
    cplot_columns = ['VEHICLE CLASS', 'TRANSMISSION', 'FUEL']
    for column in cplot_columns:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=column)
    
    plt.show()

def visualising_outliers(df):
    #outliers
    columns = ['YEAR', 'ENGINE SIZE', 'CYLINDERS', 'FUEL CONSUMPTION', 'EMISSIONS']
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df[columns])
    plt.title('OUTLIERS')
    plt.show()

def print_corelation_m(df):
    #corelation matrix
    columns = ['YEAR', 'ENGINE SIZE', 'CYLINDERS', 'FUEL CONSUMPTION', 'EMISSIONS']
    plt.figure(figsize=(10, 10))
    matrix = df[columns].corr()

    sns.heatmap(matrix, cmap="crest")
    plt.title('CORRELATION MATRIX')
    plt.show()

def print_scatter_plots(df):
    #scatter plots
    columns = ['YEAR', 'ENGINE SIZE', 'CYLINDERS', 'EMISSIONS']

    for column in columns:
        plt.figure(figsize=(8, 4))
        sns.scatterplot(data=df, x = column, y = 'FUEL CONSUMPTION')
        plt.show()

df = pd.read_csv('initial_data.csv')

df = drop_columns(df)

#simulating missing values
df.loc[df.sample(frac = 0.01).index, 'FUEL'] = np.nan
df.loc[df.sample(frac = 0.02).index, 'EMISSIONS'] = np.nan

initial_row_number = len(df)
empty_count = df.isnull().sum().sum()

df = repair_data(df)

#adding IMPACT column
avrg = df['EMISSIONS'].mean() * df['ENGINE SIZE'].mean()
impact_score = df['EMISSIONS'] * df['ENGINE SIZE']
df['IMPACT'] = ['HIGH' if x else 'LOW' for x in impact_score > avrg]

print_stats(df, initial_row_number, empty_count)
print_distribution(df)
visualising_outliers(df)
print_corelation_m(df)
print_scatter_plots(df)