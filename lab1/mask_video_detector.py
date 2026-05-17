import os
import datetime
import cv2
import numpy as np
"""
mask_video_detector.py
Утилита для детекции областей на видео по HSV-маске.

Основные шаги:
- чтение видеофайла
- конвертация кадра в HSV и построение маски
- поиск контуров и отрисовка прямоугольников вокруг найденных объектов
- (опционально) сохранение обработанного видео в папку output

Комментарии ниже поясняют назначение функций и ключевых операций.
"""

def create_disease_mask(hsv):
    """
    Создает комбинированную маску для всех типов болезней растений
    """
    # Каждая пара lower/upper определяет диапазон HSV для конкретного типа пятен
    # Маска для белых/светлых пятен
    white_spots_lower = np.array([0, 0, 200])
    white_spots_upper = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, white_spots_lower, white_spots_upper)
    
    # Маска для светло-желтых пятен болезни
    yellow_light_lower = np.array([15, 30, 150])
    yellow_light_upper = np.array([40, 150, 255])
    yellow_light_mask = cv2.inRange(hsv, yellow_light_lower, yellow_light_upper)
    
    # Маска для желто-коричневых пятен болезни
    yellow_brown_lower = np.array([10, 40, 80])
    yellow_brown_upper = np.array([35, 180, 220])
    yellow_brown_mask = cv2.inRange(hsv, yellow_brown_lower, yellow_brown_upper)
    
    # Маска для черных/темных пятен болезни
    dark_spots_lower = np.array([0, 0, 0])
    dark_spots_upper = np.array([180, 255, 80])
    dark_mask = cv2.inRange(hsv, dark_spots_lower, dark_spots_upper)
    
    # Маска для коричневых участков
    brown_lower = np.array([5, 50, 60])
    brown_upper = np.array([25, 200, 180])
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
    
    # Объединяем все маски
    combined_mask = cv2.bitwise_or(white_mask, yellow_light_mask)
    combined_mask = cv2.bitwise_or(combined_mask, yellow_brown_mask)
    combined_mask = cv2.bitwise_or(combined_mask, dark_mask)
    combined_mask = cv2.bitwise_or(combined_mask, brown_mask)
    
    # Применяем морфологические операции для очистки маски
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return combined_mask

def detect_objects_in_video(video_path, lower_bound=None, upper_bound=None, save_output=False, use_disease_mask=False):
    """
    Обнаруживает объекты в видео по HSV маске
    
    Args:
        video_path (str): Путь к видеофайлу
        lower_bound (np.array): Нижний предел HSV (для одиночной маски)
        upper_bound (np.array): Верхний предел HSV (для одиночной маски)
        save_output (bool): Сохранять ли обработанное видео
        use_disease_mask (bool): Использовать ли комбинированную маску для болезней
    """
    
    # Открываем видеофайл для постраничного чтения
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {video_path}")
        return
    
    # Получаем параметры видео (fps, ширина, высота, общее число кадров)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Видео: {video_path}")
    print(f"Разрешение: {width}x{height}, FPS: {fps}, Всего кадров: {total_frames}")
    
    # Настроим окно
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Detected Objects', cv2.WINDOW_NORMAL)
    
    # Подготовка записи обработанного видео, если пользователь выбрал сохранение
    out = None
    saved_path = None
    if save_output:
        os.makedirs('output_video', exist_ok=True)
        base = os.path.splitext(os.path.basename(video_path))[0]
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join('output_video', f"{base}_processed_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        saved_path = out_path
    
    frame_count = 0
    
    # Проходим по всем кадрам видео
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Конвертация кадра в HSV для последующей цветовой фильтрации
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Создаем маску
        if use_disease_mask:
            mask = create_disease_mask(hsv)
        else:
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # Морфологические операции для улучшения маски
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Применяем маску к исходному изображению
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Нахождение контуров на бинарной маске
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Рисуем найденные контуры и ограничивающие прямоугольники на копии кадра
        detected = frame.copy()
        
        object_count = 0
        # Проходим по всем контурам и фильтруем шум по площади
        for contour in contours:
            area = cv2.contourArea(contour)
            # Фильтруем по площади (можно настроить)
            if area > 100:
                cv2.drawContours(detected, [contour], 0, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(detected, (x, y), (x + w, y + h), (0, 0, 255), 2)
                object_count += 1
        
        # Добавляем текст с информацией
        cv2.putText(detected, f'Objects: {object_count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(detected, f'Frame: {frame_count}/{total_frames}', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Отображаем результаты
        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Detected Objects', detected)
        
        # Если включено сохранение — записываем аннотированный кадр в файл
        if save_output and out:
            out.write(detected)
        
        # Нажмите 'q' для выхода, 'p' для паузы
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)  # Пауза до нажатия любой клавиши
    
    # Освобождаем ресурсы
    cap.release()
    if out:
        out.release()
    
    cv2.destroyAllWindows()
    print(f"Обработано {frame_count} кадров")
    # Выводим путь к сохранённому видео (если было сохранение)
    if save_output and saved_path:
        print(f"Видео сохранено в '{saved_path}'")


def main():
    print("=" * 50)
    print("Детекция объектов в видео по маске HSV")
    print("=" * 50)
    
    # Ввод пути к видео
    video_path = input("\nВведите путь к видеофайлу: ").strip()
    
    # Проверяем существование файла
    import os
    if not os.path.exists(video_path):
        print(f"Ошибка: файл не найден '{video_path}'")
        return
    
    # Выбор режима детектирования
    print("\nВыберите режим детектирования:")
    print("1. Обнаружение по одиночной маске HSV")
    print("2. Обнаружение всех типов болезней (комбинированная маска)")
    
    mode_choice = input("Ваш выбор (1-2): ").strip()
    
    use_disease_mask = mode_choice == '2'
    
    if not use_disease_mask:
        # Выбор режима ввода параметров маски
        print("\nВыберите вариант маски:")
        print("1. Маска для белых пятен")
        print("2. Маска для желтых пятен")
        print("3. Ввести собственные параметры HSV")
        
        choice = input("Ваш выбор (1-3): ").strip()
        
        if choice == '1':
            # Маска для белых/светлых пятен
            lower = np.array([0, 0, 200])
            upper = np.array([180, 50, 255])
            print("Использована маска для белых пятен")
        
        elif choice == '2':
            # Маска для желтых пятен
            lower = np.array([15, 30, 150])
            upper = np.array([40, 150, 255])
            print("Использована маска для желтых пятен")
        
        elif choice == '3':
            print("\nВведите параметры маски HSV:")
            try:
                h_min = int(input("H Min (0-180): "))
                h_max = int(input("H Max (0-180): "))
                s_min = int(input("S Min (0-255): "))
                s_max = int(input("S Max (0-255): "))
                v_min = int(input("V Min (0-255): "))
                v_max = int(input("V Max (0-255): "))
                
                lower = np.array([h_min, s_min, v_min])
                upper = np.array([h_max, s_max, v_max])
            except ValueError:
                print("Ошибка ввода! Используется маска по умолчанию")
                lower = np.array([0, 0, 200])
                upper = np.array([180, 50, 255])
        
        else:
            print("Неверный выбор. Используется маска по умолчанию")
            lower = np.array([0, 0, 200])
            upper = np.array([180, 50, 255])
    else:
        print("Используется комбинированная маска для детектирования болезней:")
        print("  - Белые/светлые пятна (мозаика вируса)")
        print("  - Светло-желтые пятна")
        print("  - Желто-коричневые пятна")
        print("  - Черные/темные пятна")
        print("  - Коричневые участки")
        lower = None
        upper = None
    
    # Вопрос о сохранении видео
    save = input("\nСохранить обработанное видео? (y/n): ").strip().lower() == 'y'
    
    # Запускаем обработку видео
    detect_objects_in_video(video_path, lower, upper, save_output=save, use_disease_mask=use_disease_mask)


if __name__ == "__main__":
    main()
