import cv2

img = cv2.imread('./images/aaa.png')

gray = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 200)

contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
total = 0
for cont in contours:
        #сглаживание и определение количества углов
        sm = cv2.arcLength(cont, True)
        apd = cv2.approxPolyDP(cont, 0.02*sm, True)
        #выделение контуров
        if len(apd) == 4:
            total += 1
            cv2.drawContours(img, [apd], -1, (0,255,0), 4)
print('количество фигур = ' + str(total))
cv2.imshow('result.jpg', img)
cv2.waitKey(0)