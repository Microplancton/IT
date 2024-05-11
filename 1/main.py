import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    print(data.head(10))

    print("Размерность данных:", data.shape)
    print("Количество пустых значений в каждом столбце:")
    print(data.isnull().sum())
    print("Типы данных:")
    print(data.dtypes)

    data = data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

    data['Age'] = data['Age'].fillna(data['Age'].median())

    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    return data

titanic_data = load_and_prepare_data('titanic.csv')
print(titanic_data.head(10))