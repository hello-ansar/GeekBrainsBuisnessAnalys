import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import acf               # Инструмент, который поможет посчитать автокорреляционный
from statsmodels.graphics.tsaplots import plot_acf      # модель и её отрисовать

from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')

import missingno as miss        # Поможет делать разведовательные данные

features = pd.read_csv("Lesson_2_features.csv")
sales = pd.read_csv("Lesson_2_sales.csv")
stores = pd.read_csv("Lesson_2_stores.csv")

print(features.head())

# miss.matrix(features)   # карта пропусков
# plt.show()

print(features.describe())

'''
ЗАПОЛНЕНИЕ NaN - ов
'''
features["CPI"] = features["CPI"].fillna(features["CPI"].median())
features["Unemployment"] = features["Unemployment"].fillna(features["Unemployment"].median())
features["Temperature"] = (features["Temperature"] - 32) * 5./9.


print(sales.head())

print(sales.describe())

# sns.distplot(sales["Weekly_Sales"])

# plt.show()

print(sales[sales["Weekly_Sales"] >= 200000].head())

print(sales.info())

holidays_factor, types = sales["IsHoliday"].factorize()  # Преобразование True False в Единички и ноль
sales["IsHoliday"] = holidays_factor

print()

print(stores.info())
print(stores.head())


'''
ОБЪЕДЕНЕНИЕ ДАННЫХ
'''
df = pd.merge(sales, features, on=['Store', 'Date', 'IsHoliday'], how='left')
df = pd.merge(df, stores, on=['Store'], how='left')
df['Date'] = pd.to_datetime(df['Date'])

print(df.info())

new_df = df[['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
             'Type', 'Size']]


sales_month = new_df.groupby(df['Date'].dt.month).agg({'Weekly_Sales': 'sum'})

plt.figure(figsize=(12, 6))
sns.barplot(x=sales_month.index, y=sales_month.Weekly_Sales)
plt.xlabel('Месяц')
plt.ylabel('Продажи')

plt.show()




