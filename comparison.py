import cv2
import imutils
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Download image
imgA = cv2.imread('image/ethalon.jpg')
imgA = cv2.resize(imgA, (870, 580))
imgB = cv2.imread('image/exemplar.jpg')
imgB = cv2.resize(imgB, (870, 580))

# Gray scale conversion
grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

# Find difference
(similar, diff) = ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")

# Applying a noise threshold manually
thresh_value = 40  # This value must be selected experimentally
thresh = cv2.threshold(diff, thresh_value, 255, cv2.THRESH_BINARY_INV)[1]

# Application of morphological operations cleaning up the threshold
kernel = np.ones((3, 3), np.uint8)  # May be changed if you may need to reduce the kernel size
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Find contours
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

# Every contour is drawn in red color
for contour in contours:
    # Minimum area of the contour
    if cv2.contourArea(contour) > 500:  # May be changed if you may need to reduce the area
        # Calculate the bounding rectangle of the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        # Draw the rectangle in the image
        cv2.rectangle(imgB, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Display the similarity percentage
cv2.putText(imgB, f"Similarity: {similar * 100:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 0, 0), 2)

# Show the images in a new window
result = np.hstack((imgA, imgB))
cv2.imshow('Output', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
