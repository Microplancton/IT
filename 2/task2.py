import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('titanic.csv')

survived = df['Survived']
gender = df['Sex'] 

survived_by_gender = gender.groupby(survived).value_counts()

plt.pie(survived_by_gender, labels=survived_by_gender.index, autopct='%1.1f%%')
plt.title('Распределение выживших по полу')
plt.show()