import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('titanic.csv')

age = df['Age']
fare = df['Fare']
survived = df['Survived']

plt.scatter(age, fare, c=survived, cmap='viridis', alpha=0.5)
plt.xlabel('Возраст')
plt.ylabel('Цена билета')
plt.title('Анализ взаимосвязи между возрастом и ценой билета')
plt.colorbar(label='Выживание')
plt.show()