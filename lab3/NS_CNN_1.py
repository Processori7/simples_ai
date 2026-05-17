import os  # модуль для работы с путями и файлами
import cv2  # OpenCV для загрузки и обработки изображений
import numpy as np  # для числовых массивов
from sklearn.model_selection import train_test_split  # разделение на train/test
from tensorflow.keras.models import Sequential  # простая последовательная модель
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # слои нейросети
from tensorflow.keras.utils import to_categorical  # one-hot кодирование
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # аугментация изображений
import matplotlib.pyplot as plt  # графики

# Конфигурация
IMG_SIZE = 100  # размер изображения после изменения
# DATASET_PATH = r'C:\Users\Igor\Desktop\Zanatye_2\256_ObjectCategories'  # путь к датасету
DATASET_PATH = r'C:\256_ObjectCategories'
BATCH_SIZE = 32  # размер батча
EPOCHS = 100  # количество эпох обучения


def load_images(dataset_path):
    """Загрузка изображений из двух классов и возвращение массивов данных и меток."""
    images = []  # список для хранения изображений
    labels = []  # список для хранения меток

    # Путь к папке с первым классом (бинокли)
    binoculars_path = os.path.join(dataset_path, '012.binoculars')  # полный путь к папке
    if os.path.exists(binoculars_path):  # проверяем, существует ли папка
        for filename in os.listdir(binoculars_path):  # перебираем файлы
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # фильтруем изображения по расширению
                img_path = os.path.join(binoculars_path, filename)  # полный путь к файлу
                img = cv2.imread(img_path)  # считываем изображение с диска
                if img is None:  # если изображение не прочитано
                    continue  # пропускаем некорректный файл
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # изменяем размер изображения
                img = img / 255.0  # нормализуем пиксели в диапазон [0, 1]
                images.append(img)  # добавляем в список изображений
                labels.append(0)  # метка 0 для биноклей

    # Путь к папке со вторым классом (медведи)
    bear_path = os.path.join(dataset_path, '009.bear')  # полный путь ко второму классу
    if os.path.exists(bear_path):  # проверка наличия папки
        for filename in os.listdir(bear_path):  # перебор файлов
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # проверка типа файла
                img_path = os.path.join(bear_path, filename)  # формируем путь к файлу
                img = cv2.imread(img_path)  # читаем изображение
                if img is None:  # если чтение не удалось
                    continue  # пропуск данного изображения
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # масштабируем
                img = img / 255.0  # нормализация
                images.append(img)  # сохраняем изображение
                labels.append(1)  # метка 1 для медведей

    return np.array(images), np.array(labels)  # возвращаем numpy-массивы данных и меток


print("Загрузка изображений...")  # уведомление о начале загрузки
X, y = load_images(DATASET_PATH)  # загружаем изображения и метки
print(f"Загружено {len(X)} изображений")  # вывод количества загруженных изображений

# Разбиваем на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80/20

# One-hot кодирование меток для двух классов
y_train = to_categorical(y_train, 2)  # преобразование y_train
y_test = to_categorical(y_test, 2)  # преобразование y_test

# Построение сверточной нейронной сети
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),  # сверточный слой 1
    MaxPooling2D((2, 2)),  # пулинг 1

    Conv2D(64, (3, 3), activation='relu'),  # сверточный слой 2
    MaxPooling2D((2, 2)),  # пулинг 2

    Conv2D(128, (3, 3), activation='relu'),  # сверточный слой 3
    MaxPooling2D((2, 2)),  # пулинг 3

    Flatten(),  # разворачивание вектора признаков
    Dense(128, activation='relu'),  # полносвязный слой
    Dropout(0.5),  # регуляризация Dropout
    Dense(2, activation='softmax')  # выходной слой для двух классов
])

# Компиляция модели с оптимизатором Adam и функцией потерь
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])  # метрика точности

# Создание генератора аугментации данных
train_datagen = ImageDataGenerator(
    rotation_range=20,  # случайное вращение
    width_shift_range=0.2,  # сдвиг по ширине
    height_shift_range=0.2,  # сдвиг по высоте
    shear_range=0.2,  # сдвиг
    zoom_range=0.2,  # масштабирование
    horizontal_flip=True,  # отражение по горизонтали
    fill_mode='nearest'  # режим заполнения
)

# Генератор для подачи данных в модель
train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)  # поток данных

# Тренировка модели
history = model.fit(
    train_generator,  # обучающий генератор
    steps_per_epoch=len(X_train) // BATCH_SIZE,  # шагов за эпоху
    epochs=EPOCHS,  # число эпох
    validation_data=(X_test, y_test)  # валидационные данные
)

# Оценка модели на тесте
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)  # вычисление потерь и точности
print(f"\nТочность на тестовых данных: {test_acc*100:.2f}%")  # вывод качества

# Сохранение обученной модели
model.save('bear_bin_classifier.h5')  # пишет файл модели на диск

# Визуализация графиков обучения
plt.figure(figsize=(12, 5))  # размеры фигуры
plt.subplot(1, 2, 1)  # первый график
plt.plot(history.history['accuracy'], label='Точность')  # обучающая точность
plt.plot(history.history['val_accuracy'], label='Точность на валидации')  # валидационная точность
plt.title('График точности')  # заголовок
plt.legend()  # легенда

plt.subplot(1, 2, 2)  # второй график
plt.plot(history.history['loss'], label='Потери')  # обучающие потери
plt.plot(history.history['val_loss'], label='Потери на валидации')  # валидационные потери
plt.title('График потерь')  # заголовок
plt.legend()  # легенда
plt.show()  # показ графика
