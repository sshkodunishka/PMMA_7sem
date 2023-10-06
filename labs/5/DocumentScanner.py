import cv2
import numpy as np
import matplotlib.pyplot as plt

def ProcessScannedImage(image_path):
    # Загрузка изображения и изменение его размера
    image = cv2.imread(image_path)
    image_with_list_contour = image.copy()

    # Преобразование изображения в оттенки серого и обнаружение граней
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 75, 200)

    # Находим контуры и оставляем только пять самых больших
    cnts = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # Ищем контур с четырьмя точками (предполагаем, что это рамка)
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    # Отображаем контур на изображении
    if screenCnt is not None:
        cv2.drawContours(image_with_list_contour, [screenCnt], -1, (0, 255, 0), 2)

    # Если найден контур, выполняем преобразование перспективы
    if screenCnt is not None:
        # Находим матрицу преобразования перспективы
        rect = np.array(screenCnt, dtype="float32")
        dst = np.array([[0, 0], [500, 0], [500, 500], [0, 500]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (500, 500))

        # Преобразуем изображение в оттенки серого, затем применяем порог для эффекта "черно-белой бумаги"
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, warped = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Отражение по горизонтали и поворот изображения на 90 градусов влево
        warped = cv2.flip(warped, 1)
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Вывод изображений
        plt.figure(figsize=(12, 6))
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Оригинальное изображение")

        plt.subplot(132)
        plt.imshow(edges, cmap='gray')
        plt.axis('off')
        plt.title("Изображение с контурами")

        plt.subplot(133)
        plt.imshow(cv2.cvtColor(image_with_list_contour, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Рамка изображения")

        plt.figure(figsize=(6, 6), dpi=300)
        plt.imshow(warped, cmap='gray')
        plt.axis('off')
        plt.title("Отсканированное изображение")
        plt.show()