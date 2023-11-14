import cv2
import imutils
import numpy as np
from skimage.metrics import structural_similarity as ssim

# load the two input images
imgA = cv2.imread('image/ethalon.jpg')
imgA = cv2.resize(imgA, (870, 580))
imgB = cv2.imread('image/exemplar.jpg')
imgB = cv2.resize(imgB, (870, 580))

# Gray scaled images
grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

# Find the difference
(similar, diff) = ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
cv2.imshow('Difference', diff)

# Apply threshold
# You can manually adjust the threshold value here
THRESHOLD_VALUE = 0  # Adjust this value based on your specific images
MAX_VALUE = 255
thresh = cv2.threshold(diff, THRESHOLD_VALUE, MAX_VALUE, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Apply morphological operations for noise cleaning
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# cv2.imshow('Threshold', thresh)

# Find contours
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

# Loop over each contour
for contour in contours:
    if cv2.contourArea(contour) > 500:  # Increase this value to filter out smaller contours
        # Calculate bounding box
        (x, y, w, h) = cv2.boundingRect(contour)
        # Draw bounding box
        cv2.rectangle(imgB, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imgA, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.putText(imgB, f"Similarity: {similar * 100:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 0, 0), 2)

# Show final image with differences
x = np.zeros((360, 10, 3), np.uint8)
result = np.hstack((imgA, imgB))
cv2.imshow('Output', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
