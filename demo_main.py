import cv2
from phase_retrieval_GS import *
import matplotlib.pyplot as plt
import numpy as np

filename = 'images.jpg'
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
img = img.astype(float)
img = np.asarray(img, float)
max_iters = 1000
phase_mask = Ger_Sax_algo(img, max_iters)
plt.figure(1)
plt.subplot(131)
plt.imshow(img)
plt.title('Desired image')
plt.subplot(132)
plt.imshow(phase_mask)
plt.title('Phase mask')
plt.subplot(133)
recovery = np.fft.ifft2(np.exp(phase_mask * 1j))
plt.imshow(np.absolute(recovery)**2)
plt.title('Recovered image')
plt.show()


