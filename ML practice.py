import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_regression, make_classification  # генерация выборки для тестировки модели
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix  # для оценки результата работы модели

features, output, coef = make_regression(n_samples=10000,  # Кол.-во элементов
                                         n_features=5,  # Кол.-во признаков
                                         n_informative=4,  # Кол.-во информативных признаков
                                         n_targets=1,  # Кол.-во целевых признаков
                                         noise=0,  # Шум
                                         coef=True)

X = pd.DataFrame(features, columns=["Фактор_1", "Фактор_2", "Фактор_3", "Фактор_4", "Фактор_5"])
Y = pd.DataFrame(output, columns=["Результат"])

df = pd.concat([X, Y], axis=1)

classification_data, classification_labels = make_classification(n_samples=500,  # колсичество элементов
                                                                 n_features=5,  # количество признаков
                                                                 n_informative=4,  # количество информативных признаков
                                                                 n_classes=2,  # количество классов
                                                                 n_redundant=0,  # количество избыточных признаков
                                                                 random_state=23
                                                                 )  # возможность генерировать одинаковые выборки


df1 = pd.read_excel("Introduction.xlsx", engine="openpyxl")

for index, row in df1.iterrows():
    if row["Фактор_1"] >= 3:
        print(f"На строке {index} значение Фактора 1 больше или равно 3")
    elif row["Фактор_1"] < -3:
        print("break")
        break
    elif row["Фактор_1"] != 1:
        continue
    else:
        pass
        print("pass")


# ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА ДАННЫХ

print(df.describe())
'''
           Фактор_1      Фактор_2  ...      Фактор_5     Результат
count  10000.000000  10000.000000  ...  10000.000000  10000.000000   -  кол-во
mean       0.010216      0.003306  ...      0.023736      0.841229   -  среднее
std        1.005723      1.001421  ...      1.000296    107.033832   -  среднеквадратическое отклонение
min       -3.837664     -3.863415  ...     -3.762016   -415.828655   -  минимум
25%       -0.669716     -0.680849  ...     -0.655101    -71.768990   -  25% перцентили(медиана) (25% меньше -0.66)
50%        0.005963      0.005290  ...      0.022727      0.815815   -  50% перцентили(медиана) (50% меньше 0.00)
75%        0.705208      0.686541  ...      0.694289     73.815021   -  75% перцентили(медиана) (75% меньше 0.70)
max        4.075824      3.792219  ...      3.710950    382.498124   -  максимум

[8 rows x 6 columns]

Разница между 75% и 25% перцентили называется ИНТЕРКВАРТИЛЬНЫМ РАЗМАХОМ
'''


print(df.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 6 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   Фактор_1   10000 non-null  float64
 1   Фактор_2   10000 non-null  float64
 2   Фактор_3   10000 non-null  float64
 3   Фактор_4   10000 non-null  float64
 4   Фактор_5   10000 non-null  float64
 5   Результат  10000 non-null  float64
dtypes: float64(6)
memory usage: 468.9 KB
None
'''


df.iloc[1, 2] = np.NaN
df.iloc[1, 3] = np.NaN
df.iloc[2, 4] = np.NaN

print(df.head())

df["Фактор_5"] = df["Фактор_5"].fillna(0)
df["Фактор_3"] = df["Фактор_3"].fillna(df["Фактор_3"].mean())
df["Фактор_4"] = df["Фактор_4"].fillna(df["Фактор_4"].median())

print(df.head())


# ПОСТРОЕНИЕ МОДЕЛИ РЕГРЕССИИ

model = LinearRegression()
model.fit(X, Y)

print(LinearRegression())

print(model.coef_)

print(model.coef_[0][0])
print(model.coef_[0][1])
print(model.coef_[0][2])
print(model.coef_[0][3])
print(model.coef_[0][4])

print(coef)

print(model.predict([[1, 1, 1, 1, 1]]))


# ПОСТРОЕНИЕ МОДЕЛИ КЛАССИФИКАЦИИ

print("-------------------------------------------------------")

model = LogisticRegression()

model.fit(classification_data, classification_labels)

LogisticRegression()

print(model.coef_)

print(model.coef_[0][0])
print(model.coef_[0][1])
print(model.coef_[0][2])
print(model.coef_[0][3])
print(model.coef_[0][4])

print(classification_data)


print(model.predict([[1, 1, 1, 1, 1]]))











