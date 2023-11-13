import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def find_screen_contour(image):
    # Преобразование в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Размытие изображения для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Обнаружение краев
    edged = cv2.Canny(blurred, 30, 150)
    # Нахождение контуров
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Предполагаем, что экран имеет наибольший контур
    screen_contour = max(contours, key=cv2.contourArea)
    return screen_contour


def create_mask_for_screen(image, contour):
    # Создание маски для экрана
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    # Расширение маски, чтобы избежать граничных эффектов
    mask = cv2.dilate(mask, None, iterations=5)
    return mask


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    contour = find_screen_contour(image)
    mask = create_mask_for_screen(image, contour)
    screen = cv2.bitwise_and(image, image, mask=mask)
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    return screen_gray


def compare_images(imageA_path, imageB_path):
    processedA = preprocess_image(imageA_path)
    processedB = preprocess_image(imageB_path)

    # Проверка размера обработанных изображений
    if processedA.shape[0] < 7 or processedA.shape[1] < 7 or processedB.shape[0] < 7 or processedB.shape[1] < 7:
        raise ValueError("Обработанные изображения слишком малы для вычисления SSIM.")

    # Настройка размера окна для SSIM
    win_size = min(processedA.shape[0], processedA.shape[1], processedB.shape[0], processedB.shape[1])
    if win_size % 2 == 0:
        win_size -= 1

    # Вычисление SSIM и получение разницы
    score, diff = ssim(processedA, processedB, full=True, win_size=win_size)

    # Нормализация разницы для отображения
    diff = (diff * 255).astype("uint8")

    # Применение порогового фильтра для выделения различий
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Нахождение контуров различий
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Визуализация контуров на оригинальном изображении
    compare_image = cv2.imread(imageB_path)
    cv2.drawContours(compare_image, contours, -1, (0, 0, 255), 3)

    # Отображение и сохранение результатов
    cv2.imshow("Compared image with Differences", compare_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('highlighted_differences.png', compare_image)

    return score


# Пути к изображениям
imageA_path = 'image/ethalon.jpg'
imageB_path = 'image/exemplar.jpg'

# Сравнение обработанных изображений
try:
    similarity_score = compare_images(imageA_path, imageB_path)
    print(f"Схожесть изображений: {similarity_score * 100:.2f}%")
except ValueError as e:
    print(e)
