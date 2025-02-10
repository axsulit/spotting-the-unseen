import cv2
import numpy as np

# Load the image
# image_path = "datasets\celebdf-preprocessed\real\id0_0000_face_120.jpg"  # Update with your image path
# image = cv2.imread(image_path)

# Load the image
image = cv2.imread("datasets\\celebdf-preprocessed\\real\\id0_0000_face_120.jpg")


_, thresh = cv2.threshold(image[:, :, 0], 1, 255, cv2.THRESH_BINARY)

# Find contours of the non-black region
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get bounding rectangle of the largest contour
x, y, w, h = cv2.boundingRect(contours[0])

# Crop the image to the detected bounding box
cropped_image = image[y:y+h, x:x+w]
# Save or display the cropped image
cv2.imwrite("datasets\\celebdf-preprocessed-cropped\\cropped_image.jpg", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
