import cv2
import matplotlib.pyplot as plt

from draw_widths import draw_widths
from get_scale_ratio import get_pixel_length_ratio

im = "79"
im_path = f"images/{im}.jpg"
mask_path = f"masks/{im}.tiff"

sample_img = cv2.imread(im_path)
sample_mask = cv2.imread(mask_path, 0)

if sample_img.shape[:2] != (256, 256):
    sample_img = cv2.resize(sample_img, (256, 256), interpolation=cv2.INTER_AREA)
if sample_mask.shape != sample_img.shape[:2]:
    sample_mask = cv2.resize(sample_mask, sample_img.shape[:2], interpolation=cv2.INTER_AREA)
sample_img_gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)

fig, axes = plt.subplots(1, 3, figsize=(15, 10))
axes[0].imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
axes[1].imshow(sample_img_gray, cmap='gray')
axes[2].imshow(sample_mask, cmap='gray')
# plt.show()

img = sample_img.copy()
gray = sample_img_gray.copy()
mask = sample_mask.copy()

ratio = get_pixel_length_ratio(gray, mask, draw=True)
print(ratio)
draw_widths(img, mask, ratio=ratio, w_min=8, w_max=100, line_sim_thresh=1, group_y_range=5)