import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import datetime
import cv2
"""
classifier.py
Простой GUI-приложение на tkinter для классификации изображений "медведь/бинокль".
Ключевые части:
- загрузка моделей Keras
- выбор и отображение изображения
- подготовка изображения для модели и получение предсказания
- сохранение аннотированного результата в папку output

Комментарии в коде ниже поясняют назначение основных функций и блоков.
"""
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model # type: ignore
import warnings

warnings.filterwarnings('ignore')

# Конфигурация
IMG_SIZE = 100
MODELS_PATH = {
    'FFNN (Keras)': 'bear_bin_ffnn.keras',
    'CNN (H5)': 'bear_bin_classifier.h5'
}
CLASS_NAMES = ['Бинокль', 'Медведь']
CONFIDENCE_THRESHOLD = 0.50  # Порог уверенности (50%)


class BearClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Классификатор медведей")
        self.geometry("900x750")
        self.resizable(True, True)
        self.maxsize(1400, 900)

        self.models = {}
        self.current_model_name = None
        self.current_image = None
        self.current_image_path = None
        self.image_tk = None
        self.display_image = None

        self._build_ui()
        self._load_models()

    def _build_ui(self):
        """Создание интерфейса приложения"""
        # Здесь создаём виджеты GUI: панели, кнопки, холст для отображения
        # Главный фрейм
        main_frame = tk.Frame(self, padx=15, pady=15)
        main_frame.pack(fill="both", expand=True)

        # === Верхняя часть: выбор модели и изображения ===
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill="x", pady=(0, 15))

        # Выбор модели
        model_frame = tk.Frame(control_frame)
        model_frame.pack(fill="x", pady=(0, 10))

        tk.Label(model_frame, text="Выберите модель:", font=("Arial", 10, "bold")).pack(side="left")

        self.model_var = tk.StringVar(value='FFNN (Keras)')
        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=list(MODELS_PATH.keys()),
            state="readonly",
            width=20
        )
        self.model_combo.pack(side="left", padx=(10, 0))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_changed)

        # Кнопки для управления
        button_frame = tk.Frame(control_frame)
        button_frame.pack(fill="x")

        self.open_btn = tk.Button(
            button_frame,
            text="Открыть изображение",
            command=self._open_image,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10),
            padx=10,
            pady=5
        )
        self.open_btn.pack(side="left", padx=(0, 10))

        self.classify_btn = tk.Button(
            button_frame,
            text="Классифицировать",
            command=self._classify_image,
            bg="#2196F3",
            fg="white",
            font=("Arial", 10),
            padx=10,
            pady=5,
            state="disabled"
        )
        self.classify_btn.pack(side="left", padx=(0, 10))

        self.clear_btn = tk.Button(
            button_frame,
            text="Очистить",
            command=self._clear_image,
            bg="#f44336",
            fg="white",
            font=("Arial", 10),
            padx=10,
            pady=5
        )
        self.clear_btn.pack(side="left")

        # === Средняя часть: отображение изображения ===
        image_frame = tk.LabelFrame(main_frame, text="Предпросмотр изображения", font=("Arial", 12, "bold"))
        image_frame.pack(fill="both", expand=True, pady=(0, 15))

        self.canvas = tk.Canvas(
            image_frame,
            width=640,
            height=320,
            bg="#f0f0f0",
            highlightthickness=2,
            highlightbackground="#cccccc"
        )
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas.create_text(
            320, 160,
            text="Изображение не загружено",
            font=("Arial", 14),
            fill="#999999"
        )

        # === Нижняя часть: результаты ===
        result_frame = tk.LabelFrame(main_frame, text="Результаты классификации", font=("Arial", 12, "bold"))
        result_frame.pack(fill="both", expand=False)

        # Результат классификации
        self.result_label = tk.Label(
            result_frame,
            text="Результат: -",
            font=("Arial", 14, "bold"),
            fg="#2196F3"
        )
        self.result_label.pack(fill="x", padx=10, pady=10)

        # Вероятности
        self.prob_frame = tk.Frame(result_frame)
        self.prob_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.prob_labels = {}
        for class_name in CLASS_NAMES:
            prob_line = tk.Frame(self.prob_frame)
            prob_line.pack(fill="x", pady=5)

            tk.Label(
                prob_line,
                text=f"{class_name}:",
                font=("Arial", 10),
                width=12,
                anchor="w"
            ).pack(side="left")

            prob_bar = tk.Canvas(
                prob_line,
                width=350,
                height=20,
                bg="#e0e0e0",
                highlightthickness=1,
                highlightbackground="#999999"
            )
            prob_bar.pack(side="left", padx=(10, 0))
            self.prob_labels[class_name] = prob_bar

        # Статус
        self.status_label = tk.Label(
            result_frame,
            text="Статус: Готово",
            font=("Arial", 9),
            fg="#666666"
        )
        self.status_label.pack(fill="x", padx=10, pady=(0, 10))

    def _load_models(self):
        """Загрузка моделей"""
        # Проходим по словарю MODELS_PATH и пытаемся загрузить каждую модель
        self.status_label.config(text="Статус: Загрузка моделей...")
        self.update()

        for model_name, model_path in MODELS_PATH.items():
            try:
                full_path = os.path.join(os.getcwd(), model_path)
                if os.path.exists(full_path):
                    model = load_model(full_path)
                    self.models[model_name] = model
                    print(f"✓ Модель '{model_name}' загружена успешно")
                else:
                    print(f"✗ Модель '{model_name}' не найдена по пути: {full_path}")
            except Exception as e:
                print(f"✗ Ошибка при загрузке модели '{model_name}': {str(e)}")

        if not self.models:
            messagebox.showerror(
                "Ошибка",
                "Не удалось загрузить ни одну модель!\n"
                "Убедитесь, что файлы моделей находятся в папке программы."
            )
            self.status_label.config(text="Статус: Ошибка загрузки моделей")
        else:
            self.status_label.config(text="Статус: Модели загружены")

    def _on_model_changed(self, event=None):
        """Обработчик смены модели"""
        self.current_model_name = self.model_var.get()
        self._clear_results()

    # Методы ниже отвечают за работу с изображением: открыть, отобразить, классифицировать

    def _open_image(self):
        """Открытие диалога выбора изображения"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Все файлы", "*.*")
            ]
        )

        # Если пользователь выбрал файл — загружаем его в память и сохраняем путь
        if file_path:
            try:
                # Загрузить изображение
                self.current_image = cv2.imread(file_path)
                # Сохраняем путь для возможности сохранения результата позже
                self.current_image_path = file_path
                if self.current_image is None:
                    messagebox.showerror("Ошибка", "Не удалось загрузить изображение")
                    return

                # Отобразить в интерфейсе
                self._display_image(file_path)
                self.classify_btn.config(state="normal")
                self.status_label.config(text=f"Статус: Изображение загружено ({os.path.basename(file_path)})")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при загрузке: {str(e)}")

    def _display_image(self, file_path):
        """Отображение изображения на холсте"""
        # Используем PIL для удобного масштабирования и отображения в tkinter
        try:
            # Загрузить изображение для отображения
            image = Image.open(file_path)

            # Масштабировать, чтобы уместилось на холсте
            max_width, max_height = 620, 300
            image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

            # Преобразовать в PhotoImage
            self.image_tk = ImageTk.PhotoImage(image)
            self.display_image = image

            # Отобразить на холсте
            self.canvas.delete("all")
            self.canvas.create_image(
                320, 160,
                image=self.image_tk,
                anchor="center"
            )
        except Exception as e:
            print(f"Ошибка при отображении: {str(e)}")

    def _classify_image(self):
        """Классификация загруженного изображения"""
        if self.current_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        model_name = self.model_var.get()
        if model_name not in self.models:
            messagebox.showerror("Ошибка", "Выбранная модель не загружена")
            return

        try:
            self.status_label.config(text="Статус: Классификация...")
            self.update()

            # Подготовка изображения: изменение размера, нормализация
            img = cv2.resize(self.current_image, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float32') / 255.0

            # Обработать в зависимости от типа модели
            if 'FFNN' in model_name:
                # Для FFNN: преобразовать в плоский вектор (30000,)
                img_array = img.reshape(1, -1)
            else:
                # Для CNN: оставить как изображение (1, 100, 100, 3)
                img_array = np.expand_dims(img, axis=0)

            # Выполняем предсказание через загруженную модель
            model = self.models[model_name]
            predictions = model.predict(img_array, verbose=0)

            # Обработать результаты
            probabilities = predictions[0]
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx] * 100

            # Проверка порога уверенности
            if confidence < CONFIDENCE_THRESHOLD * 100:
                predicted_class = "❌ Неизвестный объект"
            else:
                predicted_class = CLASS_NAMES[predicted_class_idx]

            # Отображаем результаты в GUI
            self._display_results(predicted_class, confidence, probabilities, model_name)
            self.status_label.config(text="Статус: Классификация завершена")

            # Сохраняем аннотированное изображение в папку output (если успешно классифицировано)
            try:
                if self.current_image is not None:
                    os.makedirs('output', exist_ok=True)
                    base = os.path.splitext(os.path.basename(self.current_image_path))[0] if self.current_image_path else 'image'
                    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    out_path = os.path.join('output', f"{base}_classified_{ts}.jpg")
                    annotated = self.current_image.copy()
                    # Подпись результата
                    label = f"{predicted_class} ({confidence:.1f}%)"
                    cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imwrite(out_path, annotated)
                    print(f"Сохранено: {out_path}")
            except Exception as e:
                print(f"Ошибка сохранения: {e}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при классификации: {str(e)}")
            self.status_label.config(text="Статус: Ошибка")

    def _display_results(self, predicted_class, confidence, probabilities, model_name):
        """Отображение результатов классификации"""
        # Определить цвет в зависимости от уверенности
        if "Неизвестный" in predicted_class:
            color = "#f44336"  # Красный для неизвестного
        elif confidence > 70:
            color = "#4CAF50"  # Зеленый для высокой уверенности
        else:
            color = "#FF9800"  # Оранжевый для средней уверенности

        # Обновить результат
        self.result_label.config(
            text=f"Результат: {predicted_class} ({confidence:.1f}%)",
            fg=color
        )

        # Обновить вероятности
        for i, class_name in enumerate(CLASS_NAMES):
            prob_value = probabilities[i] * 100
            prob_bar = self.prob_labels[class_name]
            prob_bar.delete("all")

            # Рисовать полосу прогресса
            bar_width = int(prob_value * 3.5)
            
            # Выбрать цвет полосы
            if "Неизвестный" in predicted_class:
                bar_color = "#f44336"  # Красный
            else:
                bar_color = "#4CAF50" if i == np.argmax(probabilities) else "#2196F3"

            prob_bar.create_rectangle(
                0, 0, bar_width, 20,
                fill=bar_color,
                outline=bar_color
            )

            prob_bar.create_text(
                5, 10,
                text=f"{prob_value:.1f}%",
                anchor="w",
                font=("Arial", 9),
                fill="white"
            )

    def _clear_image(self):
        """Очистка интерфейса"""
        self.current_image = None
        self.canvas.delete("all")
        self.canvas.create_text(
            320, 160,
            text="Изображение не загружено",
            font=("Arial", 14),
            fill="#999999"
        )
        self._clear_results()
        self.classify_btn.config(state="disabled")
        self.status_label.config(text="Статус: Готово")

    def _clear_results(self):
        """Очистка результатов"""
        self.result_label.config(text="Результат: -", fg="#2196F3")
        for class_name in CLASS_NAMES:
            prob_bar = self.prob_labels[class_name]
            prob_bar.delete("all")


def main():
    app = BearClassifierApp()
    app.mainloop()


if __name__ == "__main__":
    main()
