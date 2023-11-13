import cv2
from skimage.metrics import structural_similarity as ssim


def preprocess_image(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    # Преобразование в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Фильтрация шума с помощью гауссовского размытия
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Применение адаптивного порога
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh


def compare_images(imageA, imageB):
    # Вычисление SSIM между двумя изображениями
    score, diff = ssim(imageA, imageB, full=True)
    print(f"Схожесть изображений: {score * 100:.2f}%")

    # Нормализация дельты для отображения
    diff = (diff * 255).astype("uint8")

    # Использование порога для выделения областей, где есть различия
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Наложение найденных контуров на исходное изображение
    result = cv2.cvtColor(imageA, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 1)

    return result, thresh


# Пути к изображениям
imageA_path = 'ethalon.jpg'
imageB_path = 'exemplar.jpg'

# Предобработка и сравнение изображений
preprocessed_imageA = preprocess_image(imageA_path)
preprocessed_imageB = preprocess_image(imageB_path)
result_image, threshold_image = compare_images(preprocessed_imageA, preprocessed_imageB)

# Вывод результатов
cv2.imshow("Result", result_image)
cv2.imshow("Threshold", threshold_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
