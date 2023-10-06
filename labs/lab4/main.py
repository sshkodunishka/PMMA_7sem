import cv2
import numpy as np

img = cv2.imread('images/src.jpg')

width = 800
height = 600

img = cv2.resize(img, (width, height))
gray = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)

cv2.imshow('kjklj', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ==========================
contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_contours = np.zeros(img.shape)

cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

lengths = []
areas = []

for c in contours:
    x, y, w, h = cv2.boundingRect(c)

cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

lengths.append(cv2.arcLength(c, True))
areas.append(cv2.contourArea(c))

print(f'Количество предметов: {len(contours)}')

max_length_index = np.argmax(lengths)
max_area_index = np.argmax(areas)

cv2.drawContours(img_contours, contours, max_length_index, (255, 255, 0), 3)
cv2.drawContours(img_contours, contours, max_area_index, (255, 0, 0), 3)

cv2.imshow('Contours', img_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ======================

for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    # Рисование прямоугольника вокруг каждой фигуры
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow('Rectangles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
