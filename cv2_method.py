import cv2
import numpy as np

# Загружаем изображения
image1 = cv2.imread('ethalon.jpg', 0)  # 0 означает загрузку в градациях серого
image2 = cv2.imread('exemplar.jpg', 0)

# Инициализируем ORB детектор
orb = cv2.ORB_create()

# Находим ключевые точки и дескрипторы с помощью ORB
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# Создаем BFMatcher объект
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Проводим матчинг дескрипторов
matches = bf.match(descriptors1, descriptors2)

# Сортируем матчи по расстоянию (лучшие матчи первые)
matches = sorted(matches, key=lambda x: x.distance)

# Рисуем первые 10 матчей
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=2)

# Показываем изображение
cv2.imshow('Matches', matched_image)
cv2.waitKey()
cv2.destroyAllWindows()
