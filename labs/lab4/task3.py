import cv2
import numpy as np

img_lines = cv2.imread('./images/lines.jpg')
img_circles = cv2.imread('./images/c.jpg')

desired_width = 800
desired_height = 600

img_lines = cv2.resize(img_lines, (desired_width, desired_height))
img_circles = cv2.resize(img_circles, (desired_width, desired_height))

gray_lines = cv2.Canny(cv2.cvtColor(img_lines, cv2.COLOR_BGR2GRAY), 100, 200)
gray_circles = cv2.Canny(cv2.cvtColor(img_circles, cv2.COLOR_BGR2GRAY), 100, 200)

lines = cv2.HoughLines(gray_lines, 1, np.pi / 180, 150)

if lines is not None:
    img_lines_drawn = np.zeros((desired_height, desired_width, 3), dtype=np.uint8)

    for line in lines:
        rho, theta = line[0]

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img_lines_drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Lines', img_lines_drawn)
else:
    print("Линии не найдены.")

circles = cv2.HoughCircles(gray_circles, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

img_circles_drawn = np.zeros((desired_height, desired_width, 3), dtype=np.uint8)

for circle in circles[0]:
    x, y, r = circle

    cv2.circle(img_circles_drawn, (round(x), round(y)), round(r), (0, 255, 0), 2)

print(f'Количество окружностей: {len(circles[0])}')

cv2.imshow('Circles', img_circles_drawn)
cv2.waitKey(0)
cv2.destroyAllWindows()
