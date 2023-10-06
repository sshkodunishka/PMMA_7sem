import cv2
import matplotlib.pyplot as plt

# Загрузка изображения с диска
image = cv2.imread('./images/cat.jpg')

# Проверка на успешную загрузку изображения
if image is None:
    print('Не удалось загрузить изображение.')
else:
    # Отображение исходного изображения
    cv2.imshow('Image', image)

    # Преобразование в оттенки серого (одноканальное)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray image', gray_image)

    _, binary_threshold = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    cv2.imshow('Black and white', binary_threshold)

    adaptive_threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imshow('Binary)', adaptive_threshold)

    cv2.imwrite('bin_image.jpg', adaptive_threshold)
    cv2.imwrite('black_white_image.jpg', binary_threshold)
    cv2.imwrite('gray_image.jpg', gray_image)

    # Ожидание нажатия клавиши и закрытие окон
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    dark_image = cv2.imread('./images/city.jpg', cv2.IMREAD_GRAYSCALE)
    hist = cv2.calcHist([dark_image], [0], None, [256], [0, 256])
    normal_image = cv2.equalizeHist(dark_image)
    equalized_hist = cv2.calcHist([normal_image], [0], None, [256], [0, 256])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title('Исходная гистограмма')
    plt.plot(hist)

    # Отображение гистограммы выровненного изображения
    plt.subplot(1, 2, 2)
    plt.title('Гистограмма после выравнивания')
    plt.plot(equalized_hist)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))

    # Исходное изображение
    plt.subplot(1, 2, 1)
    plt.title('Исходное изображение')
    plt.imshow(dark_image, cmap='gray')

    # Выровненное изображение
    plt.subplot(1, 2, 2)
    plt.title('Выровненное изображение')
    plt.imshow(normal_image, cmap='gray')

    plt.tight_layout()
    plt.show()