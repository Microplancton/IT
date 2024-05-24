import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Загрузка данных
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Нормализация данных
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Преобразование меток в формат one-hot
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Создание модели
model = Sequential() # Последовательные слои
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) # Сверточный слой с функцией извлечения
model.add(MaxPooling2D(pool_size=(2, 2))) # Уменьшение размерности изображений
model.add(Dropout(0.25)) # Предотвращение переобучения 
model.add(Flatten()) # Данные в одномерный вектор
model.add(Dense(128, activation='relu')) # Полносвязные слои
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) # Слой с функцией классификации

# Компиляция модели
model.compile(loss=tf.keras.losses.categorical_crossentropy, # Функция потерь 
              optimizer=tf.keras.optimizers.Adam(), 
              metrics=['accuracy']) # Оценка производительности

# Обучение модели
model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(x_test, y_test))

# Оценка модели
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
