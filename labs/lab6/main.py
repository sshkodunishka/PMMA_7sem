import cv2


face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('./mouth.xml')


image = cv2.imread('./images/image-14.png')

# Преобразование изображения в оттенки серого, так как классификаторы работают с одноканальными изображениями
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Обнаружение лиц на изображении
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Для каждого обнаруженного лица
for (x, y, w, h) in faces:
    # Рисуем прямоугольник вокруг лица
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Область интереса (ROI) для лица
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]

    # Обнаружение глаз внутри области лица
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

    # Обнаружение рта внутри области лица
    mouths = mouth_cascade.detectMultiScale(roi_gray)
    for (mx, my, mw, mh) in mouths:
        cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (255, 255, 0), 2)


# Вывод результата
cv2.imshow('Image with Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Захват видеопотока с камеры (0 - индекс камеры)
cap = cv2.VideoCapture(0)

while True:
    # Захват текущего кадра с камеры
    ret, frame = cap.read()

    if not ret:
        break

    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на кадре
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Для каждого обнаруженного лица
    for (x, y, w, h) in faces:
        # Рисуем прямоугольник вокруг лица
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Область интереса (ROI) для лица
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Обнаружение глаз внутри области лица
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

        # Обнаружение рта внутри области лица
        mouths = mouth_cascade.detectMultiScale(roi_gray)
        for (mx, my, mw, mh) in mouths:
            cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (255, 255, 0), 2)

    # Отображение текущего кадра с обнаруженными лицами
    cv2.imshow('Video with Face Detection', frame)

    # Выход из цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов и закрытие окна
cap.release()
cv2.destroyAllWindows()