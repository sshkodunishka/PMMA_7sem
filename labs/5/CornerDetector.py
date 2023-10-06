import cv2
import numpy as np
import matplotlib.pyplot as plt

def CornerDetectorHarris(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Вычисление углов с помощью детектора Харриса
    corners = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)

    # Нормализация и пороговая фильтрация углов
    corners = cv2.dilate(corners, None)
    threshold = 0.01 * corners.max()
    corner_image = image.copy()
    corner_image[corners > threshold] = [255, 0, 0]  # Отмечаем углы красным цветом

    return corner_image

def CornerDetectorShiTomasi(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Вычисление углов с помощью детектора Ши-Томаси
    corners = cv2.goodFeaturesToTrack(gray_image, maxCorners=100, qualityLevel=0.01, minDistance=10)
    corners = np.int0(corners)

    # Отмечаем углы на изображении
    corner_image = image.copy()
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(corner_image, (x, y), 3, 255, -1)  # Рисуем круги вокруг углов

    return corner_image

def СornersСombined(image_path):
    harris_image = CornerDetectorHarris(image_path)
    shi_tomasi_image = CornerDetectorShiTomasi(image_path)

    # Вывод изображений в одной фигуре
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.imshow(cv2.cvtColor(harris_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Harris Corners")

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(shi_tomasi_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Shi-Tomasi Corners")

    plt.show()