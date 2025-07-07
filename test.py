import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# 画像の読み込み
img1 = cv2.imread('original.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('compressed.png', cv2.IMREAD_GRAYSCALE)

# MSEの計算
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
mse = np.mean((img1 - img2) ** 2)

# SSIMの計算
ssim_value = ssim(img1, img2)

print(f"MSE: {mse:.2f}")
print(f"SSIM: {ssim_value:.3f}")