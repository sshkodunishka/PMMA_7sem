import cv2
import numpy as np

original_image = cv2.imread('./images/simp.jpg')

# 1. Свертка изображения линейными фильтрами
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]], dtype=np.float32)
convolved_image = cv2.filter2D(original_image, -1, kernel)

cv2.imshow('convolved_image', convolved_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Сглаживание изображений
blurred_image = cv2.blur(original_image, (5, 5))
cv2.imshow('blurred_image', blurred_image)

gaussian_blurred_image = cv2.GaussianBlur(original_image, (5, 5), 0)
cv2.imshow('gaussian_blurred_image', gaussian_blurred_image)

median_blurred_image = cv2.medianBlur(original_image, 5)
cv2.imshow('median_blurred_image', median_blurred_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. Эрозия и дилатация
second_image = cv2.imread('./images/sss.jpg')

gray_image = cv2.cvtColor(second_image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)  # Простая бинаризация

kernel = np.ones((5, 5), np.uint8)
eroded_image = cv2.erode(binary_image, kernel, iterations=1)  # Эрозия
cv2.imshow('eroded_image', eroded_image)

dilated_image = cv2.dilate(binary_image, kernel, iterations=1)  # Дилатация
cv2.imshow('dilated_image', dilated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


# 4. Адаптивная бинаризация
adaptive_threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow('adaptive_threshold', adaptive_threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Сохраните результаты
cv2.imwrite('./images/convolved_image.jpg', convolved_image)
cv2.imwrite('./images/blurred_image.jpg', blurred_image)
cv2.imwrite('./images/gaussian_blurred_image.jpg', gaussian_blurred_image)
cv2.imwrite('./images/median_blurred_image.jpg', median_blurred_image)
cv2.imwrite('./images/eroded_image.jpg', eroded_image)
cv2.imwrite('./images/dilated_image.jpg', dilated_image)
cv2.imwrite('./images/adaptive_threshold_image.jpg', adaptive_threshold)
