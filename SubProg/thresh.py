import cv2
import numpy as np

file = "5.jpg"

original = cv2.imread(file)

hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
(_ret, threshold) = cv2.threshold(hsv[:, :, 1], 90, 255, cv2.THRESH_OTSU)
dist = cv2.distanceTransform(threshold, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
idx = np.argmax(dist)
y, x = np.unravel_index(idx, dist.shape)  # corner position
color = original[y, x, :]
M = cv2.moments(threshold)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

print(180 * np.arctan2(x - cX, y - cY) / np.pi, 'degrees')
print(color[::-1], 'rgb color')
