import pandas as pd
import numpy as np

df = pd.read_csv('titanic.csv')

print("Количество признаков:", df.shape[1])
print("Типы признаков:", df.dtypes)

print("Пропущенные данные:", df.isnull().sum())

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    print(f"{col}:")
    print("Среднее значение:", df[col].mean())
    print("Стандартное отклонение:", df[col].std())
    print("Минимальное значение:", df[col].min())
    print("Максимальное значение:", df[col].max())
    print()

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"{col}:")
    print(df[col].unique())
    print()