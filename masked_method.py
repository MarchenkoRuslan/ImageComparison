import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def find_screen_contour(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur an image to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Edge detection
    edged = cv2.Canny(blurred, 30, 150)
    # Finding contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # We assume that the screen has the largest outline
    screen_contour = max(contours, key=cv2.contourArea)
    return screen_contour


def create_mask_for_screen(image, contour):
    # Creating a Screen Mask
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    # Mask expansion to avoid edge effects
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

    # Checking the size of processed images
    if processedA.shape[0] < 7 or processedA.shape[1] < 7 or processedB.shape[0] < 7 or processedB.shape[1] < 7:
        raise ValueError("Processed images are too small for SSIM calculation.")

    # Setting the window size for SSIM
    win_size = min(processedA.shape[0], processedA.shape[1], processedB.shape[0], processedB.shape[1])
    if win_size % 2 == 0:
        win_size -= 1

    # Calculating SSIM and getting the difference
    score, diff = ssim(processedA, processedB, full=True, win_size=win_size)

    # Normalizing the difference for display
    diff = (diff * 255).astype("uint8")

    # Applying a threshold filter to highlight differences
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Finding the contours of differences
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Visualization of contours on the original image
    compare_image = cv2.imread(imageB_path)
    cv2.drawContours(compare_image, contours, -1, (0, 0, 255), 3)

    # Displaying and saving results
    cv2.imshow("Compared image with Differences", compare_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('highlighted_differences.png', compare_image)

    return score


# Paths to images
imageA_path = 'image/exemplar.jpg'
imageB_path = 'image/ethalon.jpg'

# Comparison of processed images
try:
    similarity_score = compare_images(imageA_path, imageB_path)
    print(f"Similarity of images: {similarity_score * 100:.2f}%")
except ValueError as e:
    print(e)
