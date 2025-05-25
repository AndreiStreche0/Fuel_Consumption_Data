import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def drop_columns(df):
    df = df.drop("HWY (L/100 km)", axis='columns')
    df = df.drop("COMB (L/100 km)", axis='columns')
    df = df.drop("COMB (mpg)", axis='columns')
    df = df.drop("MAKE", axis='columns')
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
def string_into_integers(df):
    #processing transmission column for one hot encoding
    df['TRANSMISSION'] = df['TRANSMISSION'].str.extract(r'([A-Z]+)')

    #processing vehicle class column for one hot encoding
    freq = df['VEHICLE CLASS'].value_counts()
    saved_values = freq[freq >= 1000].index
    df['VEHICLE CLASS'] = df['VEHICLE CLASS'].where(df['VEHICLE CLASS'].isin(saved_values), 'OTHER')

    #one hot encoding
    df = pd.get_dummies(df, columns=['FUEL', 'IMPACT', 'TRANSMISSION', 'VEHICLE CLASS'], drop_first=True)

    encoder = LabelEncoder()
    df['IMPACT_LOW'] = encoder.fit_transform(df['IMPACT_LOW'])
    df['FUEL_E'] = encoder.fit_transform(df['FUEL_E'])
    df['FUEL_N'] = encoder.fit_transform(df['FUEL_N'])
    df['FUEL_X'] = encoder.fit_transform(df['FUEL_X'])
    df['FUEL_Z'] = encoder.fit_transform(df['FUEL_Z'])
    df['TRANSMISSION_AM'] = encoder.fit_transform(df['TRANSMISSION_AM'])
    df['TRANSMISSION_AS'] = encoder.fit_transform(df['TRANSMISSION_AS'])
    df['TRANSMISSION_AV'] = encoder.fit_transform(df['TRANSMISSION_AV'])
    df['TRANSMISSION_M'] = encoder.fit_transform(df['TRANSMISSION_M'])
    df['VEHICLE CLASS_FULL-SIZE'] = encoder.fit_transform(df['VEHICLE CLASS_FULL-SIZE'])
    df['VEHICLE CLASS_MID-SIZE'] = encoder.fit_transform(df['VEHICLE CLASS_MID-SIZE'])
    df['VEHICLE CLASS_OTHER'] = encoder.fit_transform(df['VEHICLE CLASS_OTHER'])
    df['VEHICLE CLASS_PICKUP TRUCK - STANDARD'] = encoder.fit_transform(df['VEHICLE CLASS_PICKUP TRUCK - STANDARD'])
    df['VEHICLE CLASS_SUBCOMPACT'] = encoder.fit_transform(df['VEHICLE CLASS_SUBCOMPACT'])
    df['VEHICLE CLASS_SUV'] = encoder.fit_transform(df['VEHICLE CLASS_SUV'])

    #changing the values in MODEL column with their frequency
    df['MODEL'] = df['MODEL'].map(df['MODEL'].value_counts())
    return df

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

df = string_into_integers(df)

#data standardization
scaler = StandardScaler()
columns = ['YEAR', 'MODEL', 'ENGINE SIZE', 'CYLINDERS', 'EMISSIONS']
df[columns] = scaler.fit_transform(df[columns])

#droping target column
X = df.drop('FUEL CONSUMPTION', axis='columns')
y = df['FUEL CONSUMPTION']

#spliting data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.to_csv('training_data.csv')
X_test.to_csv('testing_data.csv')

#training
model = LinearRegression()
model.fit(X_train, y_train)

#the prediction
prediction = model.predict(X_test)

#evaluating our result
mae = mean_absolute_error(y_test, prediction)
r2 = r2_score(y_test, prediction)

print(f"Evaluation: MAE - {mae:.4f}, R^2 - {r2:.4f}")

#prediction visualization
plt.figure(figsize=(8, 4))
sns.scatterplot(x=y_test, y=prediction, color='red')
#creating the regression line
min_val = min(y_test.min(), prediction.min())
max_val = max(y_test.max(), prediction.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.xlabel('FUEL CONSUMPTION VALUES')
plt.ylabel('PREDICTIONS')
plt.show()
