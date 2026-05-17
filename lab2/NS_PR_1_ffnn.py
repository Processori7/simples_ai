import os  # работа с файловой системой
import cv2  # OpenCV для загрузки и обработки изображений
import numpy as np  # работа с массивами
from sklearn.model_selection import train_test_split  # разделение данных на обучающую и тестовую выборки
from tensorflow.keras.models import Sequential  # последовательная модель Keras
from tensorflow.keras.layers import Dense, Dropout  # полносвязные слои и dropout
from tensorflow.keras.utils import to_categorical  # one-hot кодирование меток
import matplotlib.pyplot as plt  # графики

# Конфигурация
IMG_SIZE = 100  # размер сторон изображения после ресайза
DATASET_PATH = r'C:\Users\Igor\Desktop\Zanatye_2\256_ObjectCategories'  # путь к папке с данными
BATCH_SIZE = 32  # размер пакета при обучении
EPOCHS = 100  # количество эпох обучения


def load_images(dataset_path):
    """Загрузить изображения из папок red и green и вернуть данные и метки."""
    images = []
    labels = []

    bear_path = os.path.join(dataset_path, '009.bear')
    binoculars_path = os.path.join(dataset_path, '012.binoculars')

    # Проходим по обеим папкам: red -> label=1, green -> label=0
    for color_path, label in [(bear_path, 1), (binoculars_path, 0)]:
        if not os.path.exists(color_path):
            print(f"Папка не найдена: {color_path}")
            continue

        # Перебираем файлы в папке
        for filename in os.listdir(color_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(color_path, filename)
                img = cv2.imread(img_path)  # читаем изображение
                if img is None:
                    continue  # пропускаем, если файл не удалось загрузить

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # приводим к 100x100
                img = img / 255.0  # нормализация в [0,1]
                images.append(img)
                labels.append(label)

    # Возвращаем массивы numpy
    return np.array(images), np.array(labels)


print("Загрузка изображений...")
X, y = load_images(DATASET_PATH)
print(f"Загружено {len(X)} изображений")

if len(X) == 0:
    raise RuntimeError("Нет данных для обучения. Добавьте изображения в dataset/red и dataset/green.")

# Flatten: превращаем каждое изображение (100,100,3) в вектор длины 30000
X = X.reshape(len(X), -1)

# Разделяем данные на тренировочный и тестовый набор
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot кодирование меток (0 -> [1,0], 1 -> [0,1])
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# Создаем полносвязную (feedforward) модель
model = Sequential([
    Dense(512, activation='relu', input_shape=(IMG_SIZE * IMG_SIZE * 3,)),  # входной слой
    Dropout(0.4),  # регуляризация
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # выходной слой для 2 классов
])

# Компилируем модель с оптимизатором Adam и функцией потерь categorical_crossentropy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучаем модель
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

# Оцениваем модель на тестовых данных
scores = model.evaluate(X_test, y_test, verbose=2)

# scores[0] = loss, scores[1] = accuracy
test_loss = scores[0]
test_accuracy = scores[1]
print(f"Точность на тестовом наборе: {test_accuracy*100:.2f}%")
print(f"Потери на тестовом наборе: {test_loss:.4f}")

# Сохраняем обученную модель
model.save('bear_bin_ffnn.keras')
print('Модель сохранена: bear_bin_ffnn.keras')

# Рисуем графики метрик обучения: точность и потери
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='val_accuracy', marker='o')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.title('Точность обучения')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss', marker='o')
plt.plot(history.history['val_loss'], label='val_loss', marker='o')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.title('Потери обучения')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
