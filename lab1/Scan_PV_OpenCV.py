import os
import datetime
import cv2
import numpy as np

"""
Scan_PV_OpenCV.py
Набор функций для обработки изображений растений и выделения подозрительных (больных) областей.

Основной алгоритм в process_image:
- чтение изображения
- конвертация в HSV и построение нескольких масок по цветовым диапазонам
- объединение масок и морфологическая очистка
- поиск контуров и отрисовка результатов

Скрипт в main последовательно обрабатывает примеры изображений и сохраняет результаты в папку output.
"""

def process_image(image_path):

    image = cv2.imread(image_path)
    if image is None:
        print(f"Image upload error {image_path}")
        return None

    # Применение размытия
    # (5, 5) - размер ядра размытия, 0 - стандартное отклонение по оси X

    blurred_image = cv2.GaussianBlur(image, (31, 31), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Маска для белых/светлых пятен (мозаика вируса) - выделяем по яркости (V канал)
    # В HSV: низкая насыщенность + высокая яркость = белые/светлые пятна
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
    
    # Белые маски
    white_lower = np.array([0, 0, 0])
    white_upper = np.array([50, 36, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    
    # Объединяем все частные маски в одну общую маску
    combined_mask = cv2.bitwise_or(white_mask, yellow_light_mask)
    combined_mask = cv2.bitwise_or(combined_mask, yellow_brown_mask)
    combined_mask = cv2.bitwise_or(combined_mask, dark_mask)
    combined_mask = cv2.bitwise_or(combined_mask, brown_mask)
    
    # Очищаем маску морфологическими операциями, чтобы удалить шум и замкнуть области
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Находим контуры — границы выделенных областей
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_image = image.copy()

    is_sick = False
    affected_area = 0

    # Проходим по всем контурам и фильтруем по площади и соотношению сторон
    for contour in contours:
        area = cv2.contourArea(contour)
        # Фильтруем по размеру: не слишком маленькие и не слишком большие пятна
        if area > 100 and area < 7000:  # Уменьшил нижний предел для выявления малых пятен
            x, y, w, h = cv2.boundingRect(contour)
            
            # Фильтруем по соотношению сторон (избегаем вытянутых шумов)
            if h > 0:
                aspect_ratio = w / h
                if 0.4 < aspect_ratio < 2.5:  # Реалистичное соотношение сторон для пятен
                    # Отрисовываем прямоугольник и контур для визуализации найденной области
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.drawContours(result_image, [contour], 0, (255, 0, 0), 1)
                    is_sick = True
                    affected_area += area

    # Подписываем изображение итоговой меткой (больное/здоровое) и площадью
    label = "Disease detected! Affected area: {:.0f}px".format(affected_area) if is_sick else "Healthy leaf"
    color = (0, 0, 255) if is_sick else (0, 255, 0)
    cv2.putText(result_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return result_image

def main():

    image_paths = [
        "images\\1.jpg",
        "images\\2.jpg",
        "images\\3.jpg",
        "images\\4.jpg",
        "images\\5.jpg",
        "images\\6.jpg"
    ]

    for image_path in image_paths:
        result_image = process_image(image_path)
        if result_image is not None:
            # Показать результат и сохранить в папку output с меткой времени
            cv2.imshow(f"Result - {image_path}", result_image)
            # Сохраняем результат в папку output
            try:
                os.makedirs('output', exist_ok=True)
                base = os.path.splitext(os.path.basename(image_path))[0]
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                out_path = os.path.join('output', f"{base}_processed_{ts}.jpg")
                cv2.imwrite(out_path, result_image)
                print(f"Сохранено: {out_path}")
            except Exception as e:
                print(f"Ошибка сохранения: {e}")
            cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()