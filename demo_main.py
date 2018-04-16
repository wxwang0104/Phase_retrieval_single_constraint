import cv2


filename = 'images.jpg'
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
img = img.astype(float)
