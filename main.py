import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("Introduction.xlsx", engine="openpyxl")

print(df.head(5))

sheet_name = 0  # загружаем 1 лист
'''
sheet_name = 1  # загружаем 2 лист
sheet_name = "Имя_листа"  # загружаем страничку по названию
sheet_name = [0, 3, "Имя_листа"]  # загружаем несколько страничек
sheet_name = None  # загружаем все странички
'''

header = 0  # позволяет указать номер строки, содержащей заголовки для столбцов

usecols = [0, "Фактор_2", "Результат"]
#  позволяет указать имена или номера столбцов, которых надо загрузить

index_col = 0  # позволяет указать номер столбца, содержащей индекс (усл. номер строки)

df = pd.read_excel("Introduction.xlsx", index_col=index_col, engine="openpyxl")

print(df.head(5))

# СОРТИРОВКА
# --------------------------------------------------------------------------------

print(df.sort_values(by=["Фактор_1"]).head(5))

'''
      Фактор_1  Фактор_2  Фактор_3  Фактор_4  Фактор_5   Результат
3596 -3.594852  1.172703 -1.012452  0.165743  2.133561  117.284132
5945 -3.573703  0.932177  2.154253  0.489611  1.240095  249.968454
4027 -3.462815 -0.872593 -0.609189  1.869834  0.879972   99.044712
1172 -3.305736  0.441937  0.595063  1.831923  0.438937  246.215236
6800 -3.292729  1.293112 -0.360087  0.446890 -1.120527  103.526684
'''

print(df.sort_values(by=["Фактор_1"], ascending=False).head(5))

'''
      Фактор_1  Фактор_2  Фактор_3  Фактор_4  Фактор_5   Результат
6808  3.943094 -1.806564  0.595586  0.386546 -0.586220  -97.868637
486   3.773928  1.287303  0.333202  0.154520  0.306050  141.858806
5671  3.552384 -0.636557  0.049976  2.019021  0.224507  146.150106
1630  3.385442 -2.945990  1.021181  0.125734 -0.141862 -184.941140
4142  3.310994  0.549917  0.223626  0.763258  0.465625  138.151949
'''

# --------------------------------------------------------------------------------


# ФИЛЬТРАЦИЯ
# --------------------------------------------------------------------------------

print(df[df["Фактор_1"] > 1].head(5))
'''
    Фактор_1  Фактор_2  Фактор_3  Фактор_4  Фактор_5   Результат
1   1.709854  0.284644  0.720090 -0.794908  2.288220   36.920564
4   1.341603 -1.799876  1.337460  1.411949  0.676974   63.804022
13  1.191907 -0.197981 -1.369700 -0.134115 -0.487886 -102.785216
21  2.977954 -1.664913 -1.252203 -0.395598  0.335161 -221.266239
27  1.734742 -0.102715  1.588245  1.465060 -0.226709  196.113253
'''

df1 = df[df["Фактор_1"] > 1]

# --------------------------------------------------------------------------------


# ВИЗУАЛИЗАЦИЯ
# --------------------------------------------------------------------------------

sns.histplot(df["Фактор_1"])  # гистограмма
# plt.show()
plt.plot(df["Результат"])


# --------------------------------------------------------------------------------

