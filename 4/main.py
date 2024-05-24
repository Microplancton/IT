from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
mndata = MNIST('./dir_with_mnist_data_files')



# Загрузка датасета MNIST
(x_train, y_train), (x_test, y_test) = mndata.load_data()

# Предобработка данных
x_train, x_test = x_train / 255.0, x_test / 255.0

# Создание нейронной сети
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # Входной слой
    tf.keras.layers.Dense(128, activation='relu'), # Скрытый слой
    tf.keras.layers.Dense(10, activation='softmax') # Выходной слой
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
history = model.fit(x_train, y_train, epochs=5) # fit обучает

# Оценка качества модели
test_loss, test_accuracy = model.evaluate(x_test, y_test) 
print(f'Точность модели на тестовом наборе данных: {test_accuracy}')

# Анализ результатов и стоительство графика на каждой эпохе 

plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()