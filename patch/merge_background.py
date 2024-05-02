import cv2
import numpy as np

# Load the image
image = cv2.imread('pattern.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a binary mask of black pixels using thresholding
ret, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Invert the mask
mask_inv = cv2.bitwise_not(mask)

# Apply the mask to remove black pixels
result = cv2.bitwise_and(image, image, mask=mask_inv)

# Save the resulting image
cv2.imwrite('result_image_pattern.jpg', result)