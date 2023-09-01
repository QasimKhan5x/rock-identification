import os
from glob import glob

import numpy as np
from PIL import Image

for filepath in glob(os.path.join("masks", "*.tiff")):
    # filename without tiff extension
    filename = os.path.splitext(os.path.basename(filepath))[0] + ".png"
    # open and resize image
    img = Image.open(filepath)
    # convert image to opencv format
    mask_cv2 = np.asarray(img)
    # display image with colors
    min_value, max_value = mask_cv2.min(), mask_cv2.max()
    norm_mask = (mask_cv2 - min_value) / (max_value - min_value)
    norm_mask = (norm_mask * 255).astype("uint8")
    # update image label
    image2 = Image.fromarray(norm_mask)
    image2.save(os.path.join("masks_viz", filename))
